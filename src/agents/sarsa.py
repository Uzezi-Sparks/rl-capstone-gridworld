import numpy as np

class SARSAAgent:
    """
    SARSA: On-policy TD control
    Update rule: Q(s,a) <- Q(s,a) + alpha[r + gamma * Q(s',a') - Q(s,a)]
    """

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, n_actions))

    def get_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, next_action, done):
        """SARSA update"""
        q_current = self.Q[state, action]
        if done:
            q_target = rewardss
        else:
            q_target = reward + self.gamma * self.Q[next_state, next_action]
        self.Q[state, action] += self.alpha * (q_target - q_current)

    def get_greedy_action(self, state):
        return np.argmax(self.Q[state])

    def save(self, path):
        np.save(path, self.Q)
        print(f"Q-table saved to {path}")

    def load(self, path):
        self.Q = np.load(path)
        print(f"Q-table loaded from {path}")