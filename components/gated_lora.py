from typing import Callable, Optional, Union

import torch
from torch import nn

class GatedLoRA(nn.Module):
    """
    Gated LoRA layer that only applies adaptations to MTP tokens
    """
    def __init__(self, base_layer: nn.Linear, rank: int = 128, alpha: int = 256, dropout: float = 0.1):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.randn(rank, base_layer.in_features) * 0.01) # [rank, in_features]
        self.lora_B = nn.Parameter(torch.randn(base_layer.out_features, rank) * 0.01) # [out_features, rank]
        self.dropout = nn.Dropout(dropout)
        self._freeze_base_layer()

    def _freeze_base_layer(self):
        """Freeze base layer parameters"""
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor, mtp_mask:torch.Tensor=None) -> torch.Tensor:

        base_output = self.base_layer(x)
        lora_output = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling

        # mtp_mask: (batch_size, seq_len) -> (batch_size, seq_len, 1) -> (batch_size, seq_len, output_features)
        expanded_mask = mtp_mask.unsqueeze(-1)

        masked_lora_output = expanded_mask * lora_output
        gated_output = base_output + masked_lora_output

        return gated_output


def replace_linear_layers(model, rank=128, alpha=256, dropout=0.1):
    exclude = ["embed", "lm_head", "classifier", "norm", "layernorm", "mlp"]

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Skip excluded layers
            if any(exc in name.lower() for exc in exclude):
                continue

            # Get parent and replace
            parent_names = name.split('.')[:-1]
            attr_name = name.split('.')[-1]

            parent = model
            for parent_name in parent_names:
                parent = getattr(parent, parent_name)

            # Create GatedLoRA replacement
            replacement = GatedLoRA(
                base_layer=module,
                rank=rank,
                alpha=alpha,
                dropout=dropout
            )

            setattr(parent, attr_name, replacement)
            print(f"Replaced {name}")

def copy_weights_after_gated_lora(model):
    """Copy weights after GatedLoRA replacement"""
    for name, module in model.named_modules():
        if hasattr(module, '_original_weights'):
            # Copy weights to the base layers of GatedLoRA
            if hasattr(module.q_proj, 'base_layer'):
                module.q_proj.base_layer.weight.data.copy_(module._original_weights['q_proj'])
                module.k_proj.base_layer.weight.data.copy_(module._original_weights['k_proj'])
                module.v_proj.base_layer.weight.data.copy_(module._original_weights['v_proj'])
                module.o_proj.base_layer.weight.data.copy_(module._original_weights['o_proj'])
                
                if hasattr(module, '_original_biases'):
                    if module.q_proj.base_layer.bias is not None:
                        module.q_proj.base_layer.bias.data.copy_(module._original_biases['q_proj'])
                        module.k_proj.base_layer.bias.data.copy_(module._original_biases['k_proj'])
                        module.v_proj.base_layer.bias.data.copy_(module._original_biases['v_proj'])
                        module.o_proj.base_layer.bias.data.copy_(module._original_biases['o_proj'])
            
            # Clean up temporary storage
            delattr(module, '_original_weights')
            if hasattr(module, '_original_biases'):
                delattr(module, '_original_biases')