from src.config.vision_config import SignlipVisionConfig
from src.components_vision_model.embedding import SignlipVisionEmbeddings
from src.components_vision_model.encoder import SiglipVisionEncoder
import torch.nn as nn

class SiglipVisionModel(nn.Module):
    def __init__(self, config: SignlipVisionConfig):
        super().__init__()
        self.config = config
        
        self.embeddings = SignlipVisionEmbeddings(config)
        self.encoder = SiglipVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(self, pixel_values):
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.encoder(hidden_states)
        return self.post_layernorm(hidden_states)
