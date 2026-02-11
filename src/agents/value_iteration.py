"""
Value Iteration Algorithm
"""

class ValueIteration:
    """Tabular value iteration for MDPs"""
    
    def __init__(self, env, gamma=0.9, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V = {s: 0.0 for s in env.get_all_states()}
        self.iteration_count = 0
    
    def compute_q_value(self, state, action):
        next_state = self.env.get_next_state(state, action)
        reward = self.env.get_reward(state, action, next_state)
        return reward + self.gamma * self.V[next_state]
    
    def value_iteration_step(self):
        V_new = {}
        max_delta = 0.0
        for state in self.env.get_all_states():
            if self.env.is_terminal(state):
                V_new[state] = 0.0
                continue
            q_values = [self.compute_q_value(state, a) for a in range(self.env.num_actions)]
            V_new[state] = max(q_values)
            max_delta = max(max_delta, abs(V_new[state] - self.V[state]))
        self.V = V_new
        return max_delta
    
    def run(self, max_iterations=1000):
        print("Value Iteration:")
        print(f"γ={self.gamma}, θ={self.theta}")
        for i in range(max_iterations):
            delta = self.value_iteration_step()
            if i % 10 == 0:
                print(f"Iter {i}: Δ={delta:.6f}")
            if delta < self.theta:
                print(f"Converged after {i} iterations")
                self.iteration_count = i
                break
        return self.V
    
    def extract_policy(self):
        policy = {}
        for state in self.env.get_all_states():
            if self.env.is_terminal(state):
                policy[state] = None
                continue
            q_values = [self.compute_q_value(state, a) for a in range(self.env.num_actions)]
            policy[state] = q_values.index(max(q_values))
        return policy
    
    def get_q_table(self):
        """Return Q(s,a) table - required for V1 submission"""
        Q = {}
        for state in self.env.get_all_states():
            Q[state] = {}
            for action in range(self.env.num_actions):
                Q[state][action] = self.compute_q_value(state, action)
        return Q