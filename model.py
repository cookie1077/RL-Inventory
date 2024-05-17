import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers):
        super(Actor, self).__init__()

        layers = []
        layers.append(nn.Linear(state_dim, hidden_layers[0]))
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        layers.append(nn.Linear(hidden_layers[-1], action_dim))
        self.module_list = nn.ModuleList(layers)

    def forward(self, state):
        # each entry of the action lies in (-1, 1)
        x = F.relu(self.module_list[0](state))
        for i in range(1, len(self.module_list) - 1):
            x = F.relu(self.module_list[i](x))
        x = torch.tanh(self.module_list[-1](x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers):
        super(Critic, self).__init__()

        layers = []
        layers.append(nn.Linear(state_dim + action_dim, hidden_layers[0]))
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        layers.append(nn.Linear(hidden_layers[-1], 1))
        self.module_list = nn.ModuleList(layers)

    def forward(self, state, action):
        
        x = torch.cat([state, action], dim=1)
        for i in range(len(self.module_list) - 1):
            x = F.relu(self.module_list[i](x))
        x = self.module_list[-1](x)

        return x


if __name__ == '__main__':
    Q = Actor(5, 10, hidden_layers=[400, 300])
    pi = Critic(5, 10, hidden_layers=[400, 300])

    Q_optimizer = torch.optim.Adam(Q.parameters(), lr=1e-3)
    pi_optimizer = torch.optim.Adam(pi.parameters(), lr=1e-4)

    print(Q.parameters())