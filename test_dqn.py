from src.agents.dqn import DQNAgent
from src.environment.gridworld import PursuitEvasionEnv
import numpy as np

env = PursuitEvasionEnv(stochastic_prob=0.25)
agent = DQNAgent(state_size=4, n_actions=4)

# State format: ((agent_row, agent_col), (adv_row, adv_col))
agent_pos = (0, 0)
adv_pos = (0, 4)
state = (agent_pos, adv_pos)

# Flat state for neural network
flat_state = (0, 0, 0, 4)

action = agent.get_action(flat_state)
next_state = env.get_next_state(state, action)
reward = env.get_reward(state, action, next_state)
done = env.is_terminal(state)

# Flatten next_state for storage
flat_next = (next_state[0][0], next_state[0][1], next_state[1][0], next_state[1][1])

agent.store(flat_state, action, reward, flat_next, done)

print("DQN Agent OK!")
print(f"State: {flat_state} -> Action: {action} -> Next: {flat_next}")
print(f"Reward: {reward}, Done: {done}")
print(f"Epsilon: {agent.epsilon}")
print(f"Buffer size: {len(agent.memory)}")