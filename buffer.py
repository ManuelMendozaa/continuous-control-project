import torch
import random
import numpy as np
from collections import namedtuple, deque

MDPTuple = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

class ReplayBuffer():
    def __init__(self, buffer_size, batch_size, device, seed=0):
        # Saving variables
        self._buffer_size = buffer_size
        self._batch_size = batch_size
        self._device = device
        self._random_seed = random.seed(seed)
        self._numpy_seed = np.random.seed(seed)

        # Buffer variables
        self._buffer = deque(maxlen=buffer_size)

    def __len__(self):
        return len(self._buffer)

    def save(self, state, action, reward, next_state, done):
        """ Save new experience """
        self._buffer.append(MDPTuple(state, action, reward, next_state, done))

    def sample(self):
        """ Sample a batch of random experiences from buffer """
        # Get buffer tuples
        tuples = random.sample(self._buffer, k=self._batch_size)
        # Create torch tensors from data
        states = torch.from_numpy(np.vstack([e.state for e in tuples if e is not None])).float().to(self._device)
        actions = torch.from_numpy(np.vstack([e.action for e in tuples if e is not None])).float().to(self._device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in tuples if e is not None])).float().to(self._device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in tuples if e is not None])).float().to(self._device)
        dones = torch.from_numpy(np.vstack([e.done for e in tuples if e is not None]).astype(np.uint8)).float().to(self._device)
        return (states, actions, rewards, next_states, dones)
