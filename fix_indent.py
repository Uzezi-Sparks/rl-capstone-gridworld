new_code = '''

class TDLambdaForward:
    """
    TD(Lambda) - Forward View
    Computes lambda-return as weighted average of n-step returns.
    Requires complete episode before updating.
    """
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1, lambda_=0.9):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.lambda_ = lambda_
        self.Q = __import__('numpy').zeros((n_states, n_actions))
        self.episode = []

    def get_action(self, state):
        import numpy as np
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])

    def store_transition(self, state, action, reward):
        self.episode.append((state, action, reward))

    def compute_lambda_return(self, t):
        import numpy as np
        T = len(self.episode)
        G_lambda = 0.0
        for n in range(1, T - t + 1):
            n_step_return = 0.0
            for k in range(n):
                if t + k < T:
                    n_step_return += (self.gamma ** k) * self.episode[t + k][2]
            if t + n < T:
                s_next = self.episode[t + n][0]
                a_next = self.episode[t + n][1]
                n_step_return += (self.gamma ** n) * self.Q[s_next, a_next]
            if n < T - t:
                weight = (1 - self.lambda_) * (self.lambda_ ** (n - 1))
            else:
                weight = self.lambda_ ** (n - 1)
            G_lambda += weight * n_step_return
        return G_lambda

    def update(self):
        T = len(self.episode)
        for t in range(T):
            state, action, _ = self.episode[t]
            G_lambda = self.compute_lambda_return(t)
            self.Q[state, action] += self.alpha * (G_lambda - self.Q[state, action])
        self.episode = []

    def get_greedy_action(self, state):
        import numpy as np
        return np.argmax(self.Q[state])


class SARSAnAgent:
    """
    SARSA(n) - n-step SARSA (Forward View)
    On-policy n-step TD control.
    """
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1, n=3):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n = n
        self.Q = __import__('numpy').zeros((n_states, n_actions))
        self.buffer = []

    def get_action(self, state):
        import numpy as np
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])

    def store_transition(self, state, action, reward):
        self.buffer.append((state, action, reward))

    def update(self, next_state, next_action, done):
        if len(self.buffer) < self.n and not done:
            return
        G = sum((self.gamma ** i) * self.buffer[i][2] for i in range(len(self.buffer)))
        if not done:
            G += (self.gamma ** self.n) * self.Q[next_state, next_action]
        s, a, _ = self.buffer[0]
        self.Q[s, a] += self.alpha * (G - self.Q[s, a])
        self.buffer.pop(0)

    def reset_buffer(self):
        self.buffer = []

    def get_greedy_action(self, state):
        import numpy as np
        return np.argmax(self.Q[state])
'''

with open('src/agents/td_lambda.py', 'a') as f:
    f.write(new_code)
print('Done! Classes appended.')