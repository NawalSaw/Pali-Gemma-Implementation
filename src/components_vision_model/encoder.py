from config.vision_config import SignlipVisionConfig
import torch.nn as nn

from src.components_vision_model.attention import SiglipAttention
from src.components_vision_model.mlp import SiglipMLP

class SiglipVisionEncoder(nn.Module):
    def __init__(self, config: SignlipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SignlipVisionLayer(config) for _ in range(config.num_hidden_layers)])
    
    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states

class SignlipVisionLayer(nn.Module):
    def __init__(self, config: SignlipVisionConfig):
        super().__init__()
        self.config = config
        
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.attn = SiglipAttention(config.hidden_size, config.num_attention_heads, dropout=config.attention_dropout)
        self.mlp = SiglipMLP(config)
     
    def forward(self, hidden_states):
        # shape of hidden_states: [batch_size, num_patches, hidden_size]
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states, _ = self.attn(hidden_states, hidden_states, hidden_states)
        hidden_states = residual + hidden_states

        # shape of hidden_states: [batch_size, num_patches, hidden_size]
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states