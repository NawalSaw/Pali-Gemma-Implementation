import torch.nn as nn

class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.functional.gelu
    
    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.gelu(hidden_states, approximate='tanh')
        hidden_states = self.fc2(hidden_states)
        return hidden_states