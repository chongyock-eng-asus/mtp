import torch 
import torch.nn as nn

class SamplerHead(nn.Module):
    """
    Lightweight sampler head for coherent sequence generation
    """
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # Two-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size)
        )

        # Output projection to vocabulary
        self.output_projection = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden_states: torch.Tensor, previous_embeddings: torch.Tensor) -> torch.Tensor:
        # Concatenate hidden states with previous token embeddings
        combined = torch.cat([hidden_states, previous_embeddings], dim=-1)
        features = self.mlp(combined)

        # Project to vocabulary
        logits = self.output_projection(features)

        return logits

def add_sampler_head(model, tokenizer):
    """Add sampler head to existing model"""
    model.sampler_head = SamplerHead(
        model.config.hidden_size,
        len(tokenizer)
    )