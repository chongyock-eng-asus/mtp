from utils import setup_model_and_tokenizer
from datasets import load_dataset
from mtp_configuration import MultiTokenPredictionConfig
from components.llama_attention_bias import replace_attention_layers, LlamaAttentionBiased
from components.gated_lora import replace_linear_layers
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

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom loss computation for MTP training with component tracking
        """
        # Extract inputs
        input_ids = inputs.get("input_ids")
        position_ids = inputs.get("position_ids")
        labels = inputs.get("labels")
        mtp_mask = inputs.get("mtp_mask")

        # Forward pass through the model
        outputs = model(
            input_ids=input_ids,
            mtp_mask=mtp_mask,
            output_hidden_states=True,
        )

        logits = outputs.logits
        hidden_state = outputs.hidden_states[-1]

        # Calculate component losses
        base_loss = self._calculate_base_loss(logits, labels, mtp_mask)
        sampler_loss = self._calculate_sampler_loss(model, input_ids, hidden_state, mtp_mask, labels)
        lcm_loss = self._calculate_lcm_loss(position_ids, hidden_state, mtp_mask)
        
        # Store component losses for logging
        self.loss_components['base_loss'] = base_loss.item()
        self.loss_components['sampler_loss'] = sampler_loss.item()
        self.loss_components['lcm_loss'] = lcm_loss.item()
        
        # Combine losses with weights
        total_loss = base_loss + 0.1 * sampler_loss + 0.1 * lcm_loss

        outputs = CausalLMOutputWithPast(
            loss=total_loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
        return (total_loss, outputs) if return_outputs else total_loss

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
                print("Warning: sampler_head not found in model")
                return torch.tensor(0.0, device=hidden_state.device, requires_grad=True)
                
            sampler_loss = F.cross_entropy(sampler_logits, targets)
            
            # Debug info (only log occasionally to avoid spam)
            if self.state.global_step % 50 == 0:
                print(f"🔍 Sampler debug (step {self.state.global_step}):")
                print(f"  Valid MTP positions: {valid_positions.sum().item()}")
                print(f"  Sampler logits shape: {sampler_logits.shape}")
                print(f"  Has NaN/Inf: {torch.isnan(sampler_logits).any()}/{torch.isinf(sampler_logits).any()}")
                print(f"  Sampler loss: {sampler_loss.item():.4f}")

            return sampler_loss
            
        except Exception as e:
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
            print(f"Error in LCM loss calculation: {e}")
            return torch.tensor(0.0, device=device, requires_grad=True)

class EnhancedWandbCallback(TrainerCallback):
    """Enhanced W&B callback that logs component losses"""
    
    def on_log(self, args, state, control, model=None, logs=None, trainer=None, **kwargs):
        if logs and trainer:
            # Get component losses from trainer
            component_losses = getattr(trainer, 'loss_components', {})
            
            # Create enhanced log dict
            enhanced_logs = {
                "step": state.global_step,
                "epoch": state.epoch,
                **logs
            }
            
            # Add component losses with proper prefixes
            if component_losses:
                enhanced_logs.update({
                    f"train/base_loss": component_losses.get('base_loss', 0),
                    f"train/sampler_loss": component_losses.get('sampler_loss', 0),
                    f"train/lcm_loss": component_losses.get('lcm_loss', 0),
                    f"train/total_loss": logs.get('train_loss', 0)
                })
            
            # Log to W&B
            wandb.log(enhanced_logs)

# LoRA weight saving function removed per user request

if __name__ == "__main__":
    
    # Initialize config
    mtp_config = MultiTokenPredictionConfig()

    # Initialize W&B with better organization
    run = wandb.init(
        project="mtp-training",
        name=f"mtp-{mtp_config.num_masks}masks-lr{mtp_config.learning_rate}",
        config={
            "model_basename": mtp_config.model_basename,
            "datasource": mtp_config.datasource, 
            "num_masks": mtp_config.num_masks,
            "lora_rank": mtp_config.lora_rank,
            "lora_alpha": mtp_config.lora_alpha,
            "learning_rate": mtp_config.learning_rate,
            "max_length": mtp_config.max_length,
            "num_epochs": mtp_config.num_epochs,
        },
        tags=["mtp", "lora", f"{mtp_config.num_masks}masks"]
    )

    # Model setup
    model, tokenizer = setup_model_and_tokenizer(mtp_config)
    
    # Model modifications
    replace_attention_layers(model, LlamaAttentionBiased)
    replace_linear_layers(model)
    add_sampler_head(model, tokenizer)

    print("Setting up MTP training...")
    
    # Dataset setup
    ds = load_dataset(mtp_config.datasource, split="train")
    dataset_split = ds.train_test_split(test_size=0.1, seed=42)

    train_ds = dataset_split['train'] 
    eval_ds = dataset_split['test']

    train_dataset = MultiTokenPredictionDataset(train_ds, tokenizer, mtp_config)
    eval_dataset = MultiTokenPredictionDataset(eval_ds, tokenizer, mtp_config)

    training_args = TrainingArguments(
        output_dir=mtp_config.output_dir,
        num_train_epochs=mtp_config.num_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,

        # Memory optimization
        fp16=True,
        gradient_checkpointing=False,
        dataloader_pin_memory=False,

        # Learning and scheduling
        learning_rate=mtp_config.learning_rate,
        warmup_steps=500,
        weight_decay=0.01,

        # Logging with W&B
        logging_steps=10,
        logging_dir=f"{mtp_config.output_dir}/logs",
        report_to=["wandb"],

        # Saving
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,

        # Evaluation
        eval_strategy="steps",
        eval_steps=100,
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
        callbacks=[EnhancedWandbCallback()],
    )

    # Watch model with W&B
    wandb.watch(model, log="gradients", log_freq=100)

    print("Starting training...")

    try:
        # Train the model
        trainer.train()
        
        print("Training completed successfully!")
        
        # Save final model
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
            description="Multi-Token Prediction model"
        )
        artifact.add_dir(final_model_path)
        wandb.log_artifact(artifact)
        
        print(f"Model saved to: {final_model_path}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Log error to W&B
        wandb.log({"training_error": str(e)})
        
    finally:
        # Finish W&B run
        wandb.finish()