import torch 
import math
import torch.nn as nn

from typing import Optional, Tuple

from src.gemma.gemma_model import GemmaModel
from src.components_vision_model.vision_model import SiglipVisionModel
from src.config.pali_gemma_config import PaliGemmaConfig

class KVCache:
    def __init__(self):
        self.key_cache = []
        self.value_cache = []

    def update(
        self,
        key_states,
        value_states,
        layer_idx
    ):

        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)

        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states],
                dim=2
            )

            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states],
                dim=2
            )

        return (
            self.key_cache[layer_idx],
            self.value_cache[layer_idx]
        )

    def get_seq_length(self):
        if len(self.key_cache) == 0:
            return 0

        return self.key_cache[0].shape[2]

    def num_items(self):
        return self.get_seq_length()

class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.linear(image_features)

class GemmaForCausalLM(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()

        self.config = config
        self.model = GemmaModel(config)

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def tie_weights(self):
        self.lm_head.weight = (
            self.model.embed_tokens.weight
        )

    def forward(
        self,
        input_embeds,
        attention_mask,
        position_ids,
        kv_cache=None,
    ):

        hidden_states = self.model(
            hidden_states=input_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )

        logits = self.lm_head(hidden_states)

        return logits

class PaliGemmaModel(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.vision_model = SiglipVisionModel(config.vision_config) # Vision model processes images
        self.multi_model_projection = PaliGemmaMultiModalProjector(config) # Project image features to match language model dimensions
        self.language_model = GemmaForCausalLM(config.text_config) # LM means language model

        self.pad_token_id = config.pad_token_id if config.pad_token_id is not None else -1
    
    def tie_weights(self):
        self.language_model.tie_weights()

    def _merge_input_ids_with_image_features(
        self,
        image_features,
        input_embeds,
        input_ids,
        attention_mask,
        kv_cache,
    ):

        batch_size, seq_len = input_ids.shape
        _, num_image_tokens, embed_dim = image_features.shape

        dtype = input_embeds.dtype
        device = input_embeds.device

        scaled_image_features = (
            image_features / math.sqrt(self.config.hidden_size)
        )

        final_embeddings = torch.zeros(
            batch_size,
            seq_len,
            embed_dim,
            dtype=dtype,
            device=device,
        )

        text_mask = (
            (input_ids != self.config.image_token_index)
            & (input_ids != self.pad_token_id)
        )

        image_mask = (
            input_ids == self.config.image_token_index
        )

        pad_mask = (
            input_ids == self.pad_token_id
        )

        final_embeddings = torch.where(
            text_mask.unsqueeze(-1),
            input_embeds,
            final_embeddings,
        )

        assert image_mask.sum() == (
            scaled_image_features.numel() // embed_dim
        )

        final_embeddings = final_embeddings.masked_scatter(
            image_mask.unsqueeze(-1),
            scaled_image_features.reshape(-1),
        )

        final_embeddings = torch.where(
            pad_mask.unsqueeze(-1),
            torch.zeros_like(final_embeddings),
            final_embeddings,
        )

        q_len = seq_len

        if kv_cache is None or kv_cache.get_seq_length() == 0:

            causal_mask = torch.triu(
                torch.full(
                    (q_len, q_len),
                    float("-inf"),
                    dtype=dtype,
                    device=device,
                ),
                diagonal=1,
            )

            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            padding_mask = (attention_mask[:, None, None, :] == 0)
            causal_mask = causal_mask.masked_fill(padding_mask, float("-inf"))
            
            position_ids = (
                attention_mask.cumsum(-1) - 1
            )

            position_ids = position_ids.masked_fill(
                attention_mask == 0,
                0,
            )

            # SHAPE OF CAUSAL MASK: (1, 1, seq_len, seq_len)
            # SHAPE OF POSITION IDS: (batch_size, seq_len)

        else:

            kv_len = kv_cache.get_seq_length() + q_len

            causal_mask = torch.zeros(
                batch_size,
                1,
                q_len,
                kv_len,
                dtype=dtype,
                device=device,
            )

            position_ids = (
                attention_mask.cumsum(-1)[:, -1:]
                - 1
            ) 

            # SHAPE OF CAUSAL MASK: (batch_size, 1, q_len, kv_len)
            # SHAPE OF POSITION IDS: (batch_size, 1)

        return (
            final_embeddings,
            causal_mask,
            position_ids,
        )

    def forward(
        self, 
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        # Get input embeddings
        # shape: (batch_size, seq_len, hidden_size)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        # Get vision embeddings
        # shape: (batch_size, channels, height, width) --> (batch_size, num_patches, vision_hidden_size)
        vision_embeds = self.vision_model(pixel_values.to(input_embeds.dtype))
        
        # Project vision embeddings
        # shape: (batch_size, num_patches, vision_hidden_size) --> (batch_size, num_patches, hidden_size)
        image_features = self.multi_model_projection(vision_embeds)
        
        input_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(image_features, input_embeds, input_ids, attention_mask, kv_cache)
        
        output = self.language_model(input_embeds, attention_mask=attention_mask, position_ids=position_ids, kv_cache=kv_cache)
        
        return output
        