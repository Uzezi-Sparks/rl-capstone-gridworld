import numpy as np
from collections import defaultdict

class MonteCarloAgent:
    """
    Monte Carlo Control - On-policy every-visit MC
    
    Key difference from TD: updates only at END of episode
    using complete returns, not bootstrapped estimates.
    
    Update rule:
        G = sum of discounted rewards from t to T
        Q(s,a) <- Q(s,a) + alpha * (G - Q(s,a))
    """

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Q-table
        self.Q = np.zeros((n_states, n_actions))

        # Episode memory: stores (state, action, reward) tuples
        self.episode = []

    def get_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])

    def store_transition(self, state, action, reward):
        """Store transition during episode"""
        self.episode.append((state, action, reward))

    def update(self):
        """
        MC update at end of episode.
        Computes returns backwards from T to 0.
        """
        G = 0
        visited = []

        # Traverse episode backwards
        for state, action, reward in reversed(self.episode):
            G = reward + self.gamma * G
            visited.append((state, action, G))

        # Update Q values
        for state, action, G in visited:
            self.Q[state, action] += self.alpha * (G - self.Q[state, action])

        # Clear episode memory
        self.episode = []

    def get_greedy_action(self, state):
        return np.argmax(self.Q[state])

    def save(self, path):
        np.save(path, self.Q)
        print(f"Q-table saved to {path}")

    def load(self, path):
        self.Q = np.load(path)
        print(f"Q-table loaded from {path}")