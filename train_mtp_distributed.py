from utils import setup_model_and_tokenizer
from datasets import load_dataset
from mtp_configuration import MultiTokenPredictionConfig
from components.llama_attention_bias import replace_attention_layers, LlamaAttentionBiased
from components.gated_lora import replace_linear_layers, copy_weights_after_gated_lora
from components.sampler_head import add_sampler_head
from mtp_dataset import MultiTokenPredictionDataset
import wandb
import os
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.trainer_callback import TrainerCallback
import torch.distributed as dist
import time 
class MTPTrainer(Trainer):
    """
    Custom trainer for Multi-Token Prediction with GatedLoRA and SamplerHead
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_components = {
            'base_loss': 0.0,
            'sampler_loss': 0.0, 
            'lcm_loss': 0.0
        }
        self.timing_stats = {
            'forward_pass': 0.0,
            'base_loss': 0.0,
            'sampler_loss': 0.0,
            'lcm_loss': 0.0,
            'total_compute': 0.0
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom loss computation for MTP training with component tracking
        """
        step_start = time.time()
        # Extract inputs
        input_ids = inputs.get("input_ids")
        mtp_position_ids = inputs.get("mtp_position_ids")
        labels = inputs.get("labels")
        mtp_mask = inputs.get("mtp_mask")

        forward_start = time.time()
        # Forward pass through the model
        outputs = model(
            input_ids=input_ids,
            mtp_mask=mtp_mask,
            output_hidden_states=True,
        )
        forward_time = time.time() - forward_start

        logits = outputs.logits
        hidden_state = outputs.hidden_states[-1]

        # Calculate component losses
        base_start = time.time()
        base_loss = self._calculate_base_loss(logits, labels, mtp_mask)
        base_time = time.time() - base_start

        sampler_start = time.time()
        sampler_loss = self._calculate_sampler_loss(model, input_ids, hidden_state, mtp_mask, labels)
        sampler_time = time.time() - sampler_start

        lcm_start = time.time()
        lcm_loss = self._calculate_lcm_loss(mtp_position_ids, hidden_state, mtp_mask)
        lcm_time = time.time() - lcm_start

        total_time = time.time() - step_start

        self.timing_stats = {
            'forward_pass': forward_time * 1000,  # Convert to ms
            'base_loss': base_time * 1000,
            'sampler_loss': sampler_time * 1000,
            'lcm_loss': lcm_time * 1000,
            'total_compute': total_time * 1000
        }
        
        # Store component losses for logging
        self.loss_components['base_loss'] = base_loss.item()
        self.loss_components['sampler_loss'] = sampler_loss.item()
        self.loss_components['lcm_loss'] = lcm_loss.item()
        
        # Combine losses with weights
        total_loss = base_loss + sampler_loss + lcm_loss
        
        outputs = CausalLMOutputWithPast(
            loss=total_loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
        return (total_loss, outputs) if return_outputs else total_loss

    def log(self, logs, start_time=None):
        """Override log method to include component losses"""
        # Add component losses to the logs
        if hasattr(self, 'loss_components'):
            logs.update({
                'train/base_loss': self.loss_components.get('base_loss', 0.0),
                'train/sampler_loss': self.loss_components.get('sampler_loss', 0.0),
                'train/lcm_loss': self.loss_components.get('lcm_loss', 0.0)
            })
        
        if hasattr(self, 'timing_stats'):
            logs.update({
                'timing/forward_pass_ms': self.timing_stats.get('forward_pass', 0.0),
                'timing/base_loss_ms': self.timing_stats.get('base_loss', 0.0),
                'timing/sampler_loss_ms': self.timing_stats.get('sampler_loss', 0.0),
                'timing/lcm_loss_ms': self.timing_stats.get('lcm_loss', 0.0),
                'timing/total_compute_ms': self.timing_stats.get('total_compute', 0.0)
            })
        
        # Call parent log method
        super().log(logs)

    def _calculate_base_loss(self, logits: torch.Tensor, labels: torch.Tensor, mtp_mask: torch.Tensor):
        """Calculate base causal LM loss only on MTP positions"""
        if not mtp_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
        mtp_boolean_mask = mtp_mask.bool() 
        mtp_logits = logits[mtp_boolean_mask]
        mtp_labels = labels[mtp_boolean_mask]
        base_loss = F.cross_entropy(mtp_logits, mtp_labels, ignore_index=-100)

        return base_loss

    def _calculate_sampler_loss(self, model, input_ids, hidden_state, mtp_mask, labels):
        """Calculate sampler head loss"""
        if not mtp_mask.any():
            return torch.tensor(0.0, device=hidden_state.device, requires_grad=True)
            
        try:
            mtp_boolean_mask = mtp_mask.bool() 
            embedding = model.get_input_embeddings()
            
            # Step 1: Create the "real" token sequence using labels for MTP, input_ids for NTP
            labels_fixed = torch.where(labels == -100, input_ids, labels)
            
            # Step 2: Find valid MTP positions (not first position, has valid label)
            valid_positions = mtp_boolean_mask & (labels != -100)
            valid_positions[:, 0] = False
            
            if not valid_positions.any():
                return torch.tensor(0.0, device=hidden_state.device, requires_grad=True)
            
            # Step 3: Get indices
            batch_indices, pos_indices = valid_positions.nonzero(as_tuple=True)
            
            # Step 4: Get previous tokens, current hidden states, and targets
            prev_tokens = labels_fixed[batch_indices, pos_indices - 1]
            current_hiddens = hidden_state[batch_indices, pos_indices]
            targets = labels[batch_indices, pos_indices]
            
            # Step 5: Get embeddings and calculate loss
            prev_embeddings = embedding(prev_tokens)
            
            # Access sampler head from model
            if hasattr(model, 'sampler_head'):
                sampler_logits = model.sampler_head(current_hiddens, prev_embeddings)
            else:
                # Fallback if sampler head not found
                if self.is_local_process_zero():  # Only print on main process
                    print("Warning: sampler_head not found in model")
                return torch.tensor(0.0, device=hidden_state.device, requires_grad=True)
                
            sampler_loss = F.cross_entropy(sampler_logits, targets)
            
            # Debug info (only log occasionally to avoid spam and only on main process)
            if self.state.global_step % 50 == 0 and self.is_local_process_zero():
                print(f"🔍 Sampler debug (step {self.state.global_step}):")
                print(f"  Valid MTP positions: {valid_positions.sum().item()}")
                print(f"  Sampler logits shape: {sampler_logits.shape}")
                print(f"  Has NaN/Inf: {torch.isnan(sampler_logits).any()}/{torch.isinf(sampler_logits).any()}")
                print(f"  Sampler loss: {sampler_loss.item():.4f}")

            return sampler_loss
            
        except Exception as e:
            if self.is_local_process_zero():  # Only print on main process
                print(f"Error in sampler loss calculation: {e}")
            return torch.tensor(0.0, device=hidden_state.device, requires_grad=True)

    def _calculate_lcm_loss(self, position_mask: torch.Tensor, hidden_state: torch.Tensor, mtp_mask: torch.Tensor):
        """Calculate LCM (Latent Concept Modeling) loss"""
        if position_mask is None:
            return torch.tensor(0.0, device=hidden_state.device, requires_grad=True)
            
        B, L, H = hidden_state.shape
        device = hidden_state.device
        
        total_loss = 0.0
        total_pairs = 0
        
        try:
            for batch_idx in range(B):
                # Find unique position IDs
                unique_positions = torch.unique(position_mask[batch_idx])
                
                for pos_id in unique_positions:
                    # Get all sequence positions with this position ID
                    mask = (position_mask[batch_idx] == pos_id)
                    indices = mask.nonzero().flatten()
                    
                    if len(indices) <= 1:
                        continue  # Need multiple tokens at same logical position
                    
                    # Last index is NTP (target), others are MTP (should match target)
                    ntp_idx = indices[-1]
                    mtp_indices = indices[:-1]
                    
                    # Get target hidden state (detached)
                    ntp_hidden = hidden_state[batch_idx, ntp_idx].detach()
                    
                    # Compare each MTP hidden state to the NTP target
                    for mtp_idx in mtp_indices:
                        mtp_hidden = hidden_state[batch_idx, mtp_idx]
                        total_loss += ((mtp_hidden - ntp_hidden) ** 2).mean()
                        total_pairs += 1
            
            if total_pairs == 0:
                return torch.tensor(0.0, device=device, requires_grad=True)
            
            return total_loss / total_pairs
            
        except Exception as e:
            if self.is_local_process_zero():  # Only print on main process
                print(f"Error in LCM loss calculation: {e}")
            return torch.tensor(0.0, device=device, requires_grad=True)

class EnhancedWandbCallback(TrainerCallback):
    """Enhanced W&B callback that logs component losses - only on main process"""
    
    def on_log(self, args, state, control, model=None, logs=None, trainer=None, **kwargs):
        # Only log from the main process to avoid duplicate logs
        if logs and trainer and trainer.is_local_process_zero():
            
            # Create enhanced log dict starting with existing logs
            enhanced_logs = dict(logs)
            
            # Add step and epoch info
            enhanced_logs.update({
                "step": state.global_step,
                "epoch": state.epoch,
            })
            
            # Get component losses from trainer and add them
            if hasattr(trainer, 'loss_components'):
                component_losses = trainer.loss_components
                enhanced_logs.update({
                    'train/base_loss': component_losses.get('base_loss', 0.0),
                    'train/sampler_loss': component_losses.get('sampler_loss', 0.0),
                    'train/lcm_loss': component_losses.get('lcm_loss', 0.0)
                })
            
            # Ensure we have the total loss with proper naming
            if 'train_loss' in logs:
                enhanced_logs['train/total_loss'] = logs['train_loss']
            elif 'loss' in logs:
                enhanced_logs['train/total_loss'] = logs['loss']
            
            # Log to W&B
            wandb.log(enhanced_logs, step=state.global_step)
            
            # Debug print (occasionally) to verify logging
            if state.global_step % 100 == 0:
                print(f"Logged to W&B (step {state.global_step}): {list(enhanced_logs.keys())}")


def setup_distributed():
    """Initialize distributed training if available"""
    if "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        
        # Set device
        torch.cuda.set_device(local_rank)
        
        # Initialize process group
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank
        )
        
        return local_rank, world_size, rank
    else:
        return 0, 1, 0

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def diagnose_model_corruption(model):
    """Diagnose model corruption after modifications"""
    print("\n=== MODEL DIAGNOSTIC ===")
    
    corrupted_params = []
    for name, param in model.named_parameters():
        if param.numel() == 0:
            corrupted_params.append(name)
            print(f"CORRUPTED: {name} has shape {param.shape}")
        elif torch.isnan(param).any():
            corrupted_params.append(name)
            print(f"NaN VALUES: {name}")
    
    if corrupted_params:
        print(f"Found {len(corrupted_params)} corrupted parameters")
        return False
    else:
        print("Model validation passed - no corrupted parameters found")
        return True
    
if __name__ == "__main__":
    
    # Setup distributed training
    local_rank, world_size, rank = setup_distributed()
    is_main_process = rank == 0
    
    # Initialize config
    mtp_config = MultiTokenPredictionConfig()

    # Initialize W&B only on main process to avoid conflicts
    if is_main_process:
        os.environ["WANDB_API_KEY"] = mtp_config.WANDB_API_KEY
        os.environ["WANDB_SILENT"] = "true" 
        run = wandb.init(
            project="mtp-training",
            name=f"mtp-{mtp_config.num_masks}masks-lr{mtp_config.learning_rate}-{world_size}gpus",
            config={
                "model_basename": mtp_config.model_basename,
                "datasource": mtp_config.datasource, 
                "num_masks": mtp_config.num_masks,
                "lora_rank": mtp_config.lora_rank,
                "lora_alpha": mtp_config.lora_alpha,
                "learning_rate": mtp_config.learning_rate,
                "max_length": mtp_config.max_length,
                "num_epochs": mtp_config.num_epochs,
                "world_size": world_size,
            },
            tags=["mtp", "lora", f"{mtp_config.num_masks}masks", f"{world_size}gpus"]
        )

    # Model setup
    model, tokenizer = setup_model_and_tokenizer(mtp_config)
    
    print("Model loaded, starting modifications...")

    # Model modifications
    model.config.num_masks = mtp_config.num_masks
    model.config.max_length = mtp_config.max_length
    
    replace_attention_layers(model, LlamaAttentionBiased)
    replace_linear_layers(model)  
    copy_weights_after_gated_lora(model)  # Copy weights to base layers
    add_sampler_head(model, tokenizer)

    if is_main_process:
        print("Setting up MTP training...")
        print(f"Distributed training on {world_size} GPUs")
    
    # Dataset setup
    ds = load_dataset(mtp_config.datasource, split="train")
    dataset_split = ds.train_test_split(test_size=0.1, seed=42)

    train_ds = dataset_split['train'] 
    eval_ds = dataset_split['test']

    train_dataset = MultiTokenPredictionDataset(train_ds, tokenizer, mtp_config)
    eval_dataset = MultiTokenPredictionDataset(eval_ds, tokenizer, mtp_config)

    # Adjust batch size for distributed training
    # Total effective batch size = per_device_batch_size * gradient_accumulation_steps * world_size
    # per_device_batch_size = 1
    # gradient_accumulation_steps = max(1, 8 // world_size)  # Maintain similar effective batch size

    training_args = TrainingArguments(
        output_dir=mtp_config.output_dir,
        num_train_epochs=mtp_config.num_epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,

        # Distributed training settings
        local_rank=local_rank,
        ddp_find_unused_parameters=False,
        ddp_backend="nccl",
        deepspeed="deepspeed_config.json",

        # Memory optimization
        fp16=True,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,

        # Learning and scheduling
        learning_rate=mtp_config.learning_rate,
        warmup_steps=5000,
        weight_decay=0.01,

        # Logging with W&B (only from main process)
        logging_steps=50,
        logging_dir=f"{mtp_config.output_dir}/logs",
        report_to=["wandb"] if is_main_process else [],

        # Saving (only from main process)
        save_steps=5000,
        save_total_limit=3,
        load_best_model_at_end=True,
        # Evaluation
        eval_strategy="steps",
        eval_steps=1000,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    # Initialize enhanced trainer
    trainer = MTPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=[EnhancedWandbCallback()] if is_main_process else [],
    )

    # Watch model with W&B (only on main process)
    if is_main_process:

        wandb.watch(model, log="gradients", log_freq=100)
        print("Starting training...")

    try:
        # Train the model
        trainer.train()
        
        if is_main_process:
            print("Training completed successfully!")
        
        # Save final model (only from main process)
        if is_main_process:
            final_model_path = os.path.join(mtp_config.output_dir, "final_model")
            trainer.save_model(final_model_path)
            
            # Save config
            mtp_config.save_pretrained(mtp_config.output_dir)
            
            # Final evaluation and logging
            final_metrics = trainer.evaluate()
            wandb.log({"final_eval_loss": final_metrics.get("eval_loss", 0)})
            
            # Save model artifact to W&B
            artifact = wandb.Artifact(
                name="mtp-model",
                type="model",
                description=f"Multi-Token Prediction model trained on {world_size} GPUs"
            )
            artifact.add_dir(final_model_path)
            wandb.log_artifact(artifact)
            
            print(f"Model saved to: {final_model_path}")
        
    except Exception as e:
        if is_main_process:
            print(f"Training failed with error: {e}")
            import traceback
            traceback.print_exc()
            
            # Log error to W&B
            wandb.log({"training_error": str(e)})
        
    finally:
        # Finish W&B run (only on main process)
        if is_main_process:
            wandb.finish()
        
        # Clean up distributed training
        cleanup_distributed()