from src.agents.qlearning import QLearningAgent
from src.agents.sarsa import SARSAAgent
from src.agents.td_lambda import TDLambdaAgent, SARSALambdaAgent, QLambdaAgent

print('Testing all agents...')
QLearningAgent(625, 4)
print('Q-Learning OK!')
SARSAAgent(625, 4)
print('SARSA OK!')
TDLambdaAgent(625, 4)
print('TD Lambda OK!')
SARSALambdaAgent(625, 4)
print('SARSA Lambda OK!')
QLambdaAgent(625, 4)
print('Q Lambda OK!')
print('All agents load successfully!')

from src.agents.qlearning import QLearningAgent
from src.agents.sarsa import SARSAAgent
from src.agents.td_lambda import TDLambdaAgent, SARSALambdaAgent, QLambdaAgent
from src.agents.monte_carlo import MonteCarloAgent
from src.agents.td_n import TDnAgent


print('Testing all classical RL agents...')

QLearningAgent(625, 4)
print('✅ Q-Learning')

SARSAAgent(625, 4)
print('✅ SARSA')

TDLambdaAgent(625, 4)
print('✅ TD(Lambda)')

SARSALambdaAgent(625, 4)
print('✅ SARSA(Lambda)')

QLambdaAgent(625, 4)
print('✅ Q(Lambda)')

MonteCarloAgent(625, 4)
print('✅ Monte Carlo')

TDnAgent(625, 4, n=3)
print('✅ TD(n) - n=3')

TDnAgent(625, 4, n=5)
print('✅ TD(n) - n=5')

from src.agents.td_lambda import TDLambdaForward, SARSAnAgent

TDLambdaForward(625, 4)
print('✅ TD(Lambda) Forward View')

SARSAnAgent(625, 4, n=3)
print('✅ SARSA(n) - n=3')

SARSAnAgent(625, 4, n=5)
print('✅ SARSA(n) - n=5')

print('\n🎉 All classical RL agents loaded successfully!')