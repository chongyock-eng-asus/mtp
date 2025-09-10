from typing import Callable, Optional, Union

import torch
from torch import nn
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, rotate_half, eager_attention_forward
from transformers.models.llama.configuration_llama import LlamaConfig

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.generic import check_model_inputs
from transformers.cache_utils import Cache, DynamicCache

class LlamaAttentionBiased(LlamaAttention):
    """Multi-headed attention with custom bias mask - inherits from LlamaAttention"""
    
    def __init__(self, config: LlamaConfig, layer_idx: int):
        # Initialize the parent class first - this handles all the weight initialization
        super().__init__(config, layer_idx)
        
        # Add our custom bias mask
        bias_mask = self.build_block_causal_bias(
            config.max_length, 
            getattr(config, 'num_masks', 8),  # Default to 8 if not set
            getattr(config, 'num_masks', 8), 
            "cpu"
        )
        self.register_buffer('bias_mask', bias_mask, persistent=False)

    def build_block_causal_bias(self, seq_len, block_size, num_mask, device):
        # Create position indices
        i_indices = torch.arange(seq_len, device=device).unsqueeze(1)  # [seq_len, 1]
        j_indices = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
        
        # Standard causal mask: j <= i
        causal_mask = j_indices <= i_indices
        
        # Column causal condition: j <= i AND j % (num_mask + 1) == 0
        column_causal = causal_mask & (j_indices % (num_mask + 1) == 0)
        
        # Block indices
        i_blocks = i_indices // block_size
        j_blocks = j_indices // block_size
        
        # Block causal condition: same block AND j <= i
        block_causal = (i_blocks == j_blocks) & causal_mask
        
        # Combine conditions: True = attend, False = mask out
        allowed = column_causal | block_causal
        
        return allowed.detach() 

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        mtp_mask: Optional[torch.Tensor] = None,  # Added for MTP support
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        seq_len = hidden_states.shape[1]
        
        # Get our custom bias mask and move it to the same device as hidden_states
        bias_mask = self.bias_mask[:seq_len, :seq_len].to(hidden_states.device).unsqueeze(0).unsqueeze(0)
        
        query_states = self.q_proj(hidden_states, mtp_mask=mtp_mask).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states, mtp_mask=mtp_mask).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states, mtp_mask=mtp_mask).view(hidden_shape).transpose(1, 2)

        # Apply RoPE
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle KV cache
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Apply bias mask to attention_mask
        if attention_mask is not None:
            # Combine original attention mask with our bias mask
            attention_mask = attention_mask & bias_mask.to(attention_mask.device)
        else:
            attention_mask = bias_mask.to(hidden_states.device)

        # Use the same attention interface as parent
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,  # Our modified attention mask
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output, mtp_mask=mtp_mask)

        return attn_output, attn_weights


def replace_attention_layers(model, custom_attention_class):
    """Replace attention layers while preserving all weights"""
    replaced_count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList):
            for i, layer in enumerate(module):
                if hasattr(layer, 'self_attn'):
                    try:
                        original_attn = layer.self_attn
                        
                        # Extract config and layer_idx
                        config = original_attn.config
                        layer_idx = getattr(original_attn, 'layer_idx', i)

                        # Create new attention layer
                        new_attn = custom_attention_class(config, layer_idx)
                        
                        # Copy weights directly to ensure perfect transfer
                        with torch.no_grad():
                            new_attn.q_proj.weight.copy_(original_attn.q_proj.weight)
                            new_attn.k_proj.weight.copy_(original_attn.k_proj.weight)
                            new_attn.v_proj.weight.copy_(original_attn.v_proj.weight)
                            new_attn.o_proj.weight.copy_(original_attn.o_proj.weight)
                            
                            # Copy biases if they exist
                            if original_attn.q_proj.bias is not None:
                                new_attn.q_proj.bias.copy_(original_attn.q_proj.bias)
                            if original_attn.k_proj.bias is not None:
                                new_attn.k_proj.bias.copy_(original_attn.k_proj.bias)
                            if original_attn.v_proj.bias is not None:
                                new_attn.v_proj.bias.copy_(original_attn.v_proj.bias)
                            if original_attn.o_proj.bias is not None:
                                new_attn.o_proj.bias.copy_(original_attn.o_proj.bias)

                        # Replace the attention layer
                        layer.self_attn = new_attn
                        replaced_count += 1
                        
                        print(f"✓ Replaced attention layer {i}")
                        
                    except Exception as e:
                        print(f"✗ Failed to replace attention layer {i}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
    
    print(f"Successfully replaced {replaced_count} attention layers")
    return replaced_count