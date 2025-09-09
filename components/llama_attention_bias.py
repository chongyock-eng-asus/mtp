from typing import Callable, Optional, Union

import torch
from torch import nn
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, rotate_half,eager_attention_forward
from transformers.models.llama.configuration_llama import LlamaConfig

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.generic import check_model_inputs
from transformers.cache_utils import Cache, DynamicCache

class LlamaAttentionBiased(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)


    def build_block_causal_bias(self, seq_len, block_size, num_mask, device):
            bias = torch.full((seq_len, seq_len), float("-inf"), device=device)
            for i in range(seq_len):
                for j in range(seq_len):
                    column_causal_allowed = (j <= i) and (j % (num_mask + 1) == 0)
                    i_block, j_block = i // block_size, j // block_size
                    block_causal_allowed = (i_block == j_block) and (j <= i)
                    if column_causal_allowed or block_causal_allowed:
                        bias[i, j] = 0.0
            return bias

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        mtp_mask: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        bsz, seq_len, _ = hidden_states.size()

        num_mask = self.config.num_masks  # This will work now
        block_size = num_mask + 1

        device = hidden_states.device
        dtype = hidden_states.dtype
        cache_key = (seq_len, block_size, num_mask, str(device))

        if cache_key not in self._bias_cache:  
            print(f"Cache miss: computing bias for {cache_key}")
            self._bias_cache[cache_key] = self.build_block_causal_bias(seq_len, block_size, num_mask, device=device)
        else:
            print(f"Cache hit: reusing bias for {cache_key}")

        custom_bias = self._bias_cache[cache_key]
        custom_bias = custom_bias.unsqueeze(0).unsqueeze(0)

        # Use GatedLoRA for projections
        if mtp_mask is not None:
            query_states = self.q_proj(hidden_states, mtp_mask=mtp_mask).view(hidden_shape).transpose(1, 2)
            key_states = self.k_proj(hidden_states, mtp_mask=mtp_mask).view(hidden_shape).transpose(1, 2)
            value_states = self.v_proj(hidden_states, mtp_mask=mtp_mask).view(hidden_shape).transpose(1, 2)
        else:
            # Fallback to base layer only
            query_states = self.q_proj.base_layer(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = self.k_proj.base_layer(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = self.v_proj.base_layer(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if attention_mask is not None:
                bsz, num_heads = attention_mask.shape[:2]
                custom_bias_exp = custom_bias.expand(bsz, num_heads, seq_len, seq_len).to(dtype)
                attention_mask = attention_mask.to(dtype) + custom_bias_exp
        else:
                attention_mask = custom_bias

        self.last_bias = attention_mask
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,  # Modified attention mask with bias
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output, mtp_mask=mtp_mask)

        return attn_output, attn_weights
    

# Replace all LlamaAttention layers with custom LlamaAttentionBiased version
def replace_attention_layers(model, custom_attention_class):

    shared_bias_cache = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList):  # This is the layers ModuleList
            for i, layer in enumerate(module):
                if hasattr(layer, 'self_attn'):
                    # Get the original attention config
                    original_attn = layer.self_attn
                    config = original_attn.config
                    layer_idx = getattr(original_attn, 'layer_idx', i)

                    # Create new custom attention with same config
                    new_attn = custom_attention_class(config, layer_idx)

                    # Copy weights from original to new (if you want to preserve them)
                    if hasattr(original_attn, 'q_proj') and hasattr(new_attn, 'q_proj'):
                        # Copy base layer weights to your base layers
                        new_attn.q_proj.weight.data = original_attn.q_proj.weight.data.clone()
                        new_attn.k_proj.weight.data = original_attn.k_proj.weight.data.clone()
                        new_attn.v_proj.weight.data = original_attn.v_proj.weight.data.clone()
                        new_attn.o_proj.weight.data = original_attn.o_proj.weight.data.clone()

                    new_attn._bias_cache = shared_bias_cache
                    # Replace the attention layer
                    layer.self_attn = new_attn