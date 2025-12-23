from collections import deque
import random
import numpy as np

class ReplayBuffer:
    """Simple replay buffer for storing experience tuples."""

    # Initialize deque with max length
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    # Add experience tuple to buffer
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    # Sample a batch of experiences from the buffer
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    # Get current size of the buffer
    def __len__(self):
        return len(self.buffer)