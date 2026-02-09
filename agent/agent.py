import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math

from agent.dqn_model import DQNNet
from agent.reply_buffer import ReplayBuffer

from config.train_config import NUM_EPISODES, LR, GAMMA, BATCH_SIZE, REPLAY_BUFFER_SIZE
from config.agent_config import INITIAL_EPSILON, FINAL_EPSILON, EPSILON_DECAY, DECAY_EPISODES

    

class DQNAgent:
    def __init__(
        self,
        env: gym.Env,
        n_obs: int,
        n_actions: int,
        learning_rate: float = LR,
        gamma: float = GAMMA,
        initial_epsilon: float = INITIAL_EPSILON,
        epsilon_decay: float = EPSILON_DECAY,
        final_epsilon: float = FINAL_EPSILON,
        buffer_size: int = REPLAY_BUFFER_SIZE,
    ):
        
        """Initialize a Q-Learning agent.

        Args:
            env: The training environment
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate
            gamma (discount_factor): How much to value future rewards (0-1)
        """
        self.env = env
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_step = (self.initial_epsilon - self.final_epsilon) / self.epsilon_decay
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.step_count = 0


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQNNet(n_obs, n_actions).to(self.device) # policy network
        self.target_net = DQNNet(n_obs, n_actions).to(self.device) # target network

        self.target_net.load_state_dict(self.policy_net.state_dict()) # initialize target with policy weights
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            self.optimizer,
                            mode='max',          # reward → we want to maximize it
                            factor=0.5,          # half of the LR
                            patience=200,        # episodes without improvements
                            threshold=1.0,       # ignora micro-fluttuazioni
                            min_lr=1e-6
                        )
        self.buffer = ReplayBuffer(capacity=buffer_size)

    # Select next action given current state
    def select_action(self, state):
        self.step_count += 1
        #self.epsilon = self.final_epsilon + (self.initial_epsilon - self.final_epsilon) * \
                #np.exp(-1. * self.step_count / self.epsilon_decay)
        #self.epsilon_decay = INITIAL_EPSILON / (NUM_EPISODES)
        #self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
        #self.epsilon = max(self.final_epsilon, self.initial_epsilon - self.step_count / EPSILON_DECAY)
        
        if self.epsilon > self.final_epsilon:
            self.epsilon -= self.epsilon_step
            self.epsilon = max(self.epsilon, self.final_epsilon)
        

        if np.random.rand() < self.epsilon: # exploration
            return np.random.randint(0,self.n_actions)
        #return np.argmax(self.model.predict_on_batch(state)) # exploitation
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            return int(q_values.argmax().item())
    

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def update(self, batch_size=BATCH_SIZE):
        if len(self.buffer) < batch_size:
            return

        # Sample a batch of transitions
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self, episode):
        """Reduce exploration rate after each episode."""

        """
        if episode < 4000:
            self.epsilon = 1.0 - episode * (1.0 - 0.2) / 4000
        elif episode < 7000:
            self.epsilon = 0.2 - (episode - 4000) * (0.2 - 0.05) / 3000
        else:
            self.epsilon = 0.05 - (episode - 7000) * (0.05 - 0.01) / 2000
        
        epsilon_decay_episodes = int(0.98 * NUM_EPISODES)
        if episode < epsilon_decay_episodes:
            self.epsilon = self.initial_epsilon - episode * (
                (self.initial_epsilon - self.final_epsilon) / epsilon_decay_episodes
            )
        else:
            self.epsilon = self.final_epsilon
        
        EPS_DECAY_EPISODES = 15_000
        self.epsilon = max(
                        self.final_epsilon,
                        self.initial_epsilon - episode / EPS_DECAY_EPISODES * (self.initial_epsilon - self.final_epsilon)
                    )
        """
        #self.epsilon = max(self.final_epsilon, self.initial_epsilon - self.step_count / EPSILON_DECAY)




