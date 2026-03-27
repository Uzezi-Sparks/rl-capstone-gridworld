import numpy as np
from src.models.dqn_network import DQNNetwork
from src.replay_buffer.replay_buffer import ReplayBuffer

class DQNAgent:
    """
    Deep Q-Network (DQN)
    
    Key improvements over tabular Q-learning:
    - Neural network function approximation (handles large state spaces)
    - Experience replay (breaks correlation between samples)
    - Target network (stabilises training)
    
    Update rule: Q(s,a) <- Q(s,a) + alpha[r + gamma * max_a' Q_target(s',a') - Q(s,a)]
    """

    def __init__(self, state_size, n_actions, alpha=0.001, gamma=0.9,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 batch_size=32, buffer_size=10000, target_update_freq=100):

        self.state_size = state_size
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.step_count = 0

        # Online network (learns every step)
        self.online_net = DQNNetwork(state_size, n_actions, lr=alpha)

        # Target network (updates every target_update_freq steps)
        # Stabilises training by providing fixed Q targets
        self.target_net = DQNNetwork(state_size, n_actions, lr=alpha)
        self.target_net.set_weights(
            [w.copy() for w in self.online_net.get_weights()]
        )

        # Replay buffer
        self.memory = ReplayBuffer(
            max_size=buffer_size,
            algorithm='dqn',
            task='pursuit_evasion'
        )

    def encode_state(self, state):
        """
        Convert state tuple to neural network input vector.
        State: (agent_row, agent_col, adv_row, adv_col)
        Normalise to [0,1] for stable training.
        """
        return np.array(state, dtype=float) / 4.0  # grid size = 5

    def get_action(self, state):
        """Epsilon-greedy with decaying exploration"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        q_values = self.online_net.forward(self.encode_state(state))
        return np.argmax(q_values)

    def store(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.memory.add(state, action, reward, next_state, done)

    def train(self):
        """Sample batch and update online network"""
        if not self.memory.is_ready(self.batch_size):
            return None

        states, actions, rewards, next_states, dones = \
            self.memory.sample(self.batch_size)

        total_loss = 0
        for i in range(self.batch_size):
            s  = self.encode_state(states[i])
            ns = self.encode_state(next_states[i])

            # Current Q values
            q_vals = self.online_net.forward(s)

            # Target Q value
            if dones[i]:
                target = rewards[i]
            else:
                next_q = self.target_net.forward(ns)
                target = rewards[i] + self.gamma * np.max(next_q)

            # Compute loss gradient for action taken only
            loss_grad = np.zeros(self.n_actions)
            loss_grad[actions[i]] = q_vals[actions[i]] - target
            total_loss += loss_grad[actions[i]] ** 2

            self.online_net.backward(loss_grad)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.set_weights(
                [w.copy() for w in self.online_net.get_weights()]
            )

        return total_loss / self.batch_size

    def save(self, path_prefix):
        """Save online and target networks separately"""
        self.online_net.save(f"{path_prefix}_online.npy")
        self.target_net.save(f"{path_prefix}_target.npy")

    def load(self, path_prefix):
        self.online_net.load(f"{path_prefix}_online.npy")
        self.target_net.load(f"{path_prefix}_target.npy")