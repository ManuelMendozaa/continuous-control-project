import copy
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from config import OU_SIGMA, OU_THETA

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed=0):
        super(Actor, self).__init__()
        self._state_size = state_size
        self._action_size = action_size
        self._seed = torch.manual_seed(seed)

        # Arquitecture
        units_layer_1 = 400
        units_layer_2 = 300

        self.fc1 = nn.Linear(state_size, units_layer_1)
        self.bn1 = nn.BatchNorm1d(units_layer_1)
        self.fc2 = nn.Linear(units_layer_1, units_layer_2)
        self.fc3 = nn.Linear(units_layer_2, action_size)

        # Init layers
        self.init_weights()

    def init_weights(self):
        """ Initialize all weights """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """ Forward function responsible for mapping states to actions """
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))

class Critic(nn.Module):
    """ Cotinuous space Critic Model """
    def __init__(self, state_size, action_size, seed=0):
        super(Critic, self).__init__()
        self._state_size = state_size
        self._action_size = action_size
        self._seed = torch.manual_seed(seed)

        # Arquitecture
        units_layer_1 = 400
        units_layer_2 = 300

        self.fcs1 = nn.Linear(state_size, units_layer_1)
        self.bn1 = nn.BatchNorm1d(units_layer_1)
        self.fc2 = nn.Linear(units_layer_1 + action_size, units_layer_2)
        self.fc3 = nn.Linear(units_layer_2, 1)

        # Init layers
        self.init_weights()

    def init_weights(self):
        """ Initialize all weights """
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """ Forward function responsible for mapping (state, action) pairs to a Q_value """
        xs = F.relu(self.bn1(self.fcs1(state)))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class OUNoise():
    """ Ornstein-Unlenbeck process """
    def __init__(self, size, seed, mu=0, theta=OU_THETA, sigma=OU_SIGMA):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
