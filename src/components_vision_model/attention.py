import torch 
import torch.nn as nn
import math

class SiglipAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, hidden_size = hidden_states.size()
        
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        query = query.view(batch_size, seq_len, self.num_attention_heads, hidden_size // self.num_attention_heads).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_attention_heads, hidden_size // self.num_attention_heads).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_attention_heads, hidden_size // self.num_attention_heads).transpose(1, 2)
        
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(hidden_size // self.num_attention_heads)
        
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        attention_probs = nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32).to(query.dtype)
        attention_probs = nn.functional.dropout(attention_probs, p=self.dropout, training=self.training) 
        
        context_layer = torch.matmul(attention_probs, value)
        context_layer = context_layer.transpose(1, 2).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        return self.out_proj(context_layer), attention_probs