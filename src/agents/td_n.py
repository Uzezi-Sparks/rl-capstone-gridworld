import numpy as np
from collections import deque

class TDnAgent:
    """
    n-Step TD Control (Forward View)
    
    Instead of bootstrapping from next step (TD(0)) or waiting 
    for full episode (MC), uses n steps of real rewards then bootstraps.
    
    n=1  -> TD(0) (standard Q-learning/SARSA)
    n=inf -> Monte Carlo
    
    Update rule:
        G(t:t+n) = r_t + gamma*r_t+1 + ... + gamma^n * Q(s_t+n, a_t+n)
        Q(s,a)   <- Q(s,a) + alpha * (G(t:t+n) - Q(s,a))
    """

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, 
                 epsilon=0.1, n=3):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n = n  # number of steps

        self.Q = np.zeros((n_states, n_actions))

        # Buffers to store n steps of experience
        self.states  = deque(maxlen=n+1)
        self.actions = deque(maxlen=n+1)
        self.rewards = deque(maxlen=n+1)

    def get_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])

    def store_transition(self, state, action, reward):
        """Store transition in buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def update(self, done=False):
        """
        n-step TD update.
        Only updates when buffer has n steps OR episode ends.
        """
        if len(self.states) < self.n and not done:
            return  # Not enough steps yet

        # Compute n-step return G
        G = 0
        for i, r in enumerate(self.rewards):
            G += (self.gamma ** i) * r

        # Bootstrap from nth state if not terminal
        if not done and len(self.states) == self.n + 1:
            G += (self.gamma ** self.n) * np.max(self.Q[self.states[-1]])

        # Update Q for the OLDEST state/action in buffer
        s = self.states[0]
        a = self.actions[0]
        self.Q[s, a] += self.alpha * (G - self.Q[s, a])

    def reset_buffers(self):
        """Clear buffers at start of each episode"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()

    def get_greedy_action(self, state):
        return np.argmax(self.Q[state])

    def save(self, path):
        np.save(path, self.Q)
        print(f"Q-table saved to {path}")

    def load(self, path):
        self.Q = np.load(path)
        print(f"Q-table loaded from {path}")