
from src.config.vision_config import SignlipVisionConfig
import torch.nn as nn
import torch

class SignlipVisionEmbeddings(nn.Module):
    def __init__(self, config: SignlipVisionConfig):
        super().__init__() 
        self.config = config
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding="valid",
        )
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, config.hidden_size)

        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)
    
    def forward(self, pixel_values):
        # shape of pixel_values: [batch_size, num_channels, height, width]
        # shape of patch_embeddings initial: [batch_size, hidden_size, height_convoluted, width_convoluted]
        # shape of patch_embeddings final: [batch_size, num_patches, hidden_size]
        # shape of position_embeddings: [1, num_patches, hidden_size]

        patch_embeddings = self.patch_embedding(pixel_values)
        patch_embeddings = patch_embeddings.flatten(2).transpose(1, 2) 
        position_embeddings = self.position_embedding(self.position_ids[:, :patch_embeddings.size(1)]) 
        return patch_embeddings + position_embeddings
