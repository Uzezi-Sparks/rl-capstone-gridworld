import numpy as np

class TDLambdaAgent:
    """
    TD(Lambda) with Eligibility Traces - Backward View
    
    Forward view: theoretical, averages n-step returns weighted by lambda
    Backward view: practical implementation using eligibility traces
    
    Update rule:
        delta = r + gamma * V(s') - V(s)       # TD error
        e(s) = gamma * lambda * e(s) + 1        # eligibility trace
        V(s) <- V(s) + alpha * delta * e(s)     # value update
    """

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, 
                 epsilon=0.1, lambda_=0.9):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.lambda_ = lambda_

        # Q-table
        self.Q = np.zeros((n_states, n_actions))

        # Eligibility traces (same shape as Q)
        self.e = np.zeros((n_states, n_actions))

    def get_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])

    def reset_traces(self):
        """Reset eligibility traces at start of each episode"""
        self.e = np.zeros((self.n_states, self.n_actions))

    def update(self, state, action, reward, next_state, next_action, done):
        """
        Backward-view TD(Lambda) update (SARSA-Lambda)
        
        Steps:
        1. Compute TD error
        2. Increment eligibility trace for visited (s,a)
        3. Update ALL Q(s,a) weighted by their trace
        4. Decay all traces
        """
        # Step 1: TD error
        if done:
            delta = reward - self.Q[state, action]
        else:
            delta = (reward + self.gamma * self.Q[next_state, next_action] 
                    - self.Q[state, action])

        # Step 2: Increment trace for current (state, action)
        self.e[state, action] += 1

        # Step 3 & 4: Update all Q values and decay traces
        self.Q += self.alpha * delta * self.e
        self.e *= self.gamma * self.lambda_

        return delta

    def get_greedy_action(self, state):
        return np.argmax(self.Q[state])

    def save(self, path):
        np.save(path, self.Q)
        print(f"Q-table saved to {path}")

    def load(self, path):
        self.Q = np.load(path)
        print(f"Q-table loaded from {path}")


class SARSALambdaAgent(TDLambdaAgent):
    """
    SARSA(Lambda) - explicit subclass of TD(Lambda)
    On-policy control with eligibility traces
    
    Forward view:  weighted average of n-step SARSA returns
    Backward view: eligibility traces (implemented here)
    """
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9,
                 epsilon=0.1, lambda_=0.9):
        super().__init__(n_states, n_actions, alpha, gamma, epsilon, lambda_)
        print(f"SARSA(λ) initialized | lambda={lambda_}, alpha={alpha}, gamma={gamma}")


class QLambdaAgent(TDLambdaAgent):
    """
    Q(Lambda) - off-policy TD(Lambda)
    Uses max Q(s',a') instead of actual next action
    """
    def update(self, state, action, reward, next_state, next_action, done):
        """Off-policy update using max Q"""
        if done:
            delta = reward - self.Q[state, action]
        else:
            # Off-policy: use max instead of actual next action
            delta = (reward + self.gamma * np.max(self.Q[next_state])
                    - self.Q[state, action])

        self.e[state, action] += 1
        self.Q += self.alpha * delta * self.e
        self.e *= self.gamma * self.lambda_

        return delta

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
