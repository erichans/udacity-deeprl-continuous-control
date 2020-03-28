import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)
    
#def reset_parameters(layer, final_layer):
#    layer.weight.data.uniform_(-3e-3, 3e-3) if final_layer else layer.weight.data.uniform_(*hidden_init(layer))

class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_size)
        
        self.reset_parameters()
        #[reset_parameters(layer, layer == self.fc4) for layer in [self.fc1, self.fc2, self.fc3, self.fc4]]
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))
        
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

        
class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fcs1 = nn.Linear(state_size, 400)
        self.fc2 = nn.Linear(400+action_size, 300)
        self.fc3 = nn.Linear(300, 50)
        self.fc4 = nn.Linear(50, 1)
        
        self.reset_parameters()
        #[reset_parameters(layer, layer == self.fc4) for layer in [self.fcs1, self.fc2, self.fc3, self.fc4]]
        
    def forward(self, x, action):
        x = F.relu(self.fcs1(x))
        x = F.relu(self.fc2(torch.cat([x, action], dim=1)))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
    
    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)