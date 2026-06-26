import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class GemmaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([GemmaLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, hidden_states, position_ids, attention_mask=None, kv_cache=None):
        hidden_states = (
            hidden_states *
            math.sqrt(self.config.hidden_size)
        )
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, position_ids=position_ids, attention_mask=attention_mask, kv_cache=kv_cache, layer_idx=i)
        return self.norm(hidden_states)

class GemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
    
    def forward(self, hidden_states):
        gate = F.gelu(
            self.gate_proj(hidden_states),
            approximate="tanh"
        )
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.register_buffer(
            "inv_freq",
            1.0 / (
                base **
                (torch.arange(0, dim, 2).float() / dim)
            ),
            persistent=False
        )

    def forward(self, q, k, position_ids):
        freqs = torch.einsum("bt,d->btd", position_ids.float(), self.inv_freq.to(position_ids.device)) # shape --> (batch_size, seq_len, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1) # shape --> (batch_size, seq_len, dim) what it does is duplicate the frequencies and concatenate them because we need to apply the same rotation to both query and key
        cos = emb.cos().unsqueeze(1)
        sin = emb.sin().unsqueeze(1)

        # SHAPE OF Q: (batch_size, num_heads, seq_len, head_dim)
        # SHAPE OF K: (batch_size, num_heads, seq_len, head_dim)
        # SHAPE OF COS: (batch_size, 1, seq_len, head_dim)
        # SHAPE OF SIN: (batch_size, 1, seq_len, head_dim)

        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        return q, k
    
    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

class GroupedMultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_key_value_groups
        self.max_position_embeddings = config.max_position_embeddings
        self.attention_dropout = config.attention_dropout
    
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
    
    def _repeat_kv(self, hidden_states, n_rep):
        batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, kv_cache=None, layer_idx=None):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        q = q.view(q.size(0), q.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.num_key_value_heads, self.head_dim).transpose(1, 2)

        q, k = self.rotary_emb(q, k, position_ids)

        # update
        if kv_cache is not None:
            k, v = kv_cache.update(k, v, layer_idx)

        k = self._repeat_kv(k, self.num_key_value_groups)
        v = self._repeat_kv(v, self.num_key_value_groups)

        # Attention calculation        
        attention = torch.matmul(q, k.transpose(-2, -1))
        attention = attention / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attention = attention + attention_mask

        attention = torch.softmax(attention, dim=-1, dtype=torch.float32).to(q.dtype)
        attention = F.dropout(attention, self.attention_dropout, training=self.training)
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).contiguous()
        output = output.reshape(output.size(0), output.size(1), self.hidden_size)
        output = self.o_proj(output)
        
        return output
        
class GemmaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.self_attn = GroupedMultiHeadAttention(config)
        self.mlp = GemmaMLP(config)

        self.pre_attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.pre_mlp_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_mlp_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(self, hidden_states, attention_mask=None, position_ids=None, kv_cache=None, layer_idx=None):
        
        residual = hidden_states

        hidden_states = self.pre_attn_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask, position_ids=position_ids, kv_cache=kv_cache, layer_idx=layer_idx)
        hidden_states = self.post_attn_norm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states

        hidden_states = self.pre_mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_mlp_norm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
