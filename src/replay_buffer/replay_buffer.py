import numpy as np
import os
import json
from collections import deque
from datetime import datetime

class ReplayBuffer:
    """
    Experience Replay Buffer
    
    Stores transitions (s, a, r, s', done) for off-policy learning.
    Organized by algorithm, task, and policy freshness.
    
    Features:
    - Fixed max size with automatic oldest-experience replacement
    - Save/load to disk organized by algorithm and task
    - Size assertions to keep storage manageable
    - Sampling for batch learning (DQN etc.)
    """

    MAX_BUFFER_SIZE = 100000  # Hard cap

    def __init__(self, max_size=10000, algorithm='unknown', task='default'):
        assert max_size <= self.MAX_BUFFER_SIZE, \
            f"Buffer size {max_size} exceeds hard cap {self.MAX_BUFFER_SIZE}"

        self.max_size = max_size
        self.algorithm = algorithm
        self.task = task
        self.buffer = deque(maxlen=max_size)
        self.created_at = datetime.now().isoformat()

    def add(self, state, action, reward, next_state, done):
        """Add transition to buffer (auto-replaces oldest if full)"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample random batch of transitions"""
        assert len(self.buffer) >= batch_size, \
            f"Not enough experience: {len(self.buffer)} < {batch_size}"
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states      = np.array([t[0] for t in batch])
        actions     = np.array([t[1] for t in batch])
        rewards     = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones       = np.array([t[4] for t in batch])
        
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

    def is_ready(self, batch_size):
        """Check if buffer has enough experience to sample"""
        return len(self.buffer) >= batch_size

    def save(self, base_path='results/replay_data'):
        """Save buffer to disk organized by algorithm/task"""
        save_dir = os.path.join(base_path, self.algorithm, self.task)
        os.makedirs(save_dir, exist_ok=True)

        # Save transitions
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        data_path = os.path.join(save_dir, f'buffer_{timestamp}.npy')
        np.save(data_path, np.array(list(self.buffer), dtype=object))

        # Save metadata
        meta = {
            'algorithm': self.algorithm,
            'task': self.task,
            'size': len(self.buffer),
            'max_size': self.max_size,
            'created_at': self.created_at,
            'saved_at': datetime.now().isoformat(),
            'file': data_path
        }
        meta_path = os.path.join(save_dir, f'meta_{timestamp}.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"Buffer saved: {data_path} ({len(self.buffer)} transitions)")
        return data_path

    def load(self, path):
        """Load buffer from disk"""
        data = np.load(path, allow_pickle=True)
        self.buffer = deque(
            [tuple(t) for t in data], 
            maxlen=self.max_size
        )
        print(f"Buffer loaded: {path} ({len(self.buffer)} transitions)")