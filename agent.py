import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from config import *
from models import Actor, Critic, OUNoise
from buffer import ReplayBuffer

class Agent():
    """ DDPG Agent for a Continuous action space """
    def __init__(self,
                 state_size,
                 action_size,
                 device,
                 buffer_size=BUFFER_SIZE,
                 batch_size=BATCH_SIZE,
                 gamma=GAMMA,
                 tau=TAU,
                 lr=LR,
                 epsilon=EPSILON,
                 epsilon_decay=EPSILON_DECAY,
                 n_steps=N_STEPS,
                 learning_iters=LEARNING_ITERS,
                 seed=SEED):
        # Saving variables
        self._state_size = state_size          #  environment state space
        self._action_size = action_size        #  environment discrete action space
        self._lr = lr                          #  (α) learning rate
        self._gamma = gamma                    #  (γ) discount factor
        self._tau = tau                        #  (τ) soft update factor
        self._epsilon = epsilon                #  (ε) noise factor
        self._epsilon_decay = epsilon_decay    #  ε decay per training
        self._batch_size = batch_size          #  size of every sample batch
        self._buffer_size = buffer_size        #  max number of experiences to store
        self._n_steps = n_steps                #  bootstrapping from experiences for the loss
        self._device = device                  #  available processing device
        self._learning_iters = learning_iters  #  number of loops at every learning step

        # Actor Networks
        self._actor_local = Actor(state_size, action_size, seed=seed).to(device)
        self._actor_target = Actor(state_size, action_size, seed=seed).to(device)
        self._actor_optimizer = optim.Adam(self._actor_local.parameters(), lr=lr)

        # Critic Networks
        self._critic_local = Critic(state_size, action_size, seed=seed).to(device)
        self._critic_target = Critic(state_size, action_size, seed=seed).to(device)
        self._critic_optimizer = optim.Adam(self._critic_local.parameters(), lr=lr)

        # Noise process
        self._noise = OUNoise(action_size, seed)

        # Replay buffer
        self._buffer = ReplayBuffer(buffer_size, batch_size, device)

    def step(self, state, action, reward, next_state, done, t):
        self._buffer.save(state, action, reward, next_state, done)

        if len(self._buffer) > self._batch_size and t % self._n_steps == 0:
            for _ in range(self._learning_iters):
                self.learn()

    def act(self, state, add_noise=True):
        """ Agent takes the action based on the state and its actor local model """
        # Convert state into torch tensor
        state = torch.from_numpy(state).float().to(self._device)

        # Actor forward
        self._actor_local.eval()
        with torch.no_grad():
            action = self._actor_local(state).cpu().data.numpy()
        self._actor_local.train()

        # Add noise for exploration
        if add_noise:
            action += self._epsilon * self._noise.sample()

        # Keep continuous action between -1 and 1
        return np.clip(action, -1, 1)


    def learn(self):
        # Decompose buffer sample
        states, actions, rewards, next_states, dones = self._buffer.sample()

        # Critic forward
        next_actions = self._actor_target(next_states)
        Q_values_next = self._critic_target(next_states, next_actions)
        Q_values = rewards + (self._gamma * Q_values_next * (1 - dones))
        Q_values_expected = self._critic_local(states, actions)
        # Critic loss
        critic_loss = F.mse_loss(Q_values_expected, Q_values)
        # Critic backward
        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._critic_local.parameters(), 1)
        self._critic_optimizer.step()

        # Actor forward
        local_actions = self._actor_local(states)
        actor_loss = -self._critic_local(states, local_actions).mean()
        # Actor backward
        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        self._actor_optimizer.step()

        # Update target networks
        self.soft_update()

        # Update noise factor
        self._epsilon = max(self._epsilon - self._epsilon_decay, 0.02)
        self._noise.reset()

    def soft_update(self):
        """ Update the target networks with their own parameters and the local networks' """
        # Critic update
        for critic_target_param, critic_local_param in zip(self._critic_target.parameters(), self._critic_local.parameters()):
            critic_target_param.data.copy_(self._tau * critic_local_param.data + (1.0-self._tau) * critic_target_param.data)

        # Actor update
        for actor_target_param, actor_local_param in zip(self._actor_target.parameters(), self._actor_local.parameters()):
            actor_target_param.data.copy_(self._tau * actor_local_param.data + (1.0-self._tau) * actor_target_param.data)
    
    def reset(self):
        self._noise.reset()