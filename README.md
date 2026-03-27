# RL Capstone: Pursuit-Evasion GridWorld

**Author:** Uzezi Olorunmola  
**Course:** Reinforcement Learning (Spring 2026)  
**Institution:** University of North Dakota

---

## Project Overview

Reinforcement learning capstone exploring **pursuit-evasion dynamics** in gridworld environments. An agent must navigate to a goal while avoiding a pursuing adversary, requiring strategic reasoning and multi-agent coordination.

**V1 Status:** Tabular dynamic programming with scripted adversary (625-state MDP)

---

## Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/[your-username]/rl-capstone-gridworld.git
cd rl-capstone-gridworld

# Install dependencies
pip install -r requirements.txt
```

### Run V1
```bash
python src/train.py
```

**Outputs:**
- `results/v1_baseline/policy.pkl` - Optimal policy π*(s)
- `results/v1_baseline/values.pkl` - State values V*(s)
- `results/v1_baseline/q_table.pkl` - Action values Q*(s,a)
- `results/v1_baseline/gridworld.png` - Summary visualization

---

## V1 Results

**Environment:** 5×5 Pursuit-Evasion GridWorld  
**State Space:** 625 states (agent_pos × adversary_pos)  
**Algorithm:** Value Iteration (Dynamic Programming)  
**Hyperparameters:** γ=0.9, θ=10⁻⁴  
**Convergence:** 9 iterations  

See `docs/v1_proposal.md` for complete MDP justification and methodology.

---

## Why Pursuit-Evasion?

Unlike basic pathfinding, pursuit-evasion requires:
- **Strategic reasoning** about adversary behavior
- **Non-stationary environment** (opponent moves)
- **Risk management** (safety vs. efficiency tradeoff)
- **Foundation for multi-agent RL** (V3: self-play)

**Comparable complexity to:**
- Healthcare optimization (resource allocation under uncertainty)
- Game AI (adversarial decision-making)
- Robotics (dynamic obstacle avoidance)

---

## Project Structure
```
rl-capstone-gridworld/
├── README.md                      # This file
├── requirements.txt               # Dependencies
├── docs/
│   └── v1_proposal.md            # MDP justification
├── src/
│   ├── environment/
│   │   └── gridworld.py          # Pursuit-evasion MDP
│   ├── agents/
│   │   └── value_iteration.py    # DP algorithm
│   └── train.py                  # Main script
└── results/
    └── v1_baseline/              # Saved artifacts
```

---

## Development Roadmap

### ✅ V1 (Week 5) - Foundation
- [x] Pursuit-evasion MDP (625 states)
- [x] Scripted adversary patrol
- [x] Value iteration implementation
- [x] Save policy, values, Q-table

### V2 - Model-Free Learning (Next Milestone)
- Stochastic adversary behavior
- Q-learning implementation
- SARSA implementation
- Learning curves and ablations

### V3 - Advanced Topics (Future Direction)
- Self-play with learning adversary
- Partial observability
- Scaling to larger environments
- Performance analysis

Project scope and direction may evolve based on course content and research interests.
---

## Technical Details

### MDP Formulation

**State:** (agent_position, adversary_position) ∈ (5×5) × (5×5) = 625 states

**Actions:** {UP, DOWN, LEFT, RIGHT}

**Transitions:** 
- Agent: deterministic movement
- Adversary: clockwise patrol pattern

**Rewards:**
- Goal reached: +10.0
- Caught by adversary: -10.0
- Adjacent to adversary: -0.5 (proximity penalty)
- Normal step: -0.1

**Discount:** γ = 0.9

### Algorithm Performance

- Convergence: 9 iterations (θ = 10⁻⁴)
- Computation: ~10 seconds
- Policy: Successfully avoids adversary while reaching goal

---

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Littman, M. L. (1994). *Markov games as a framework for multi-agent reinforcement learning.* ICML.
- Gymnasium: https://gymnasium.farama.org/

---

## Academic Integrity

**Solo Project:** Uzezi Olorunmola

**Development Tools:**
- Python 3.13, NumPy 2.2.5, Matplotlib 3.10.1
- Git/GitHub, Cookiecutter Data Science template
- Claude AI (Anthropic) - debugging assistance and documentation review
- Full LLM conversation logs archived (available per course policy)

All code implementations personally verified and understood.

---

## V2: Model-Free Learning
**Last Updated: March 26, 2026 | Version 2 Submission**

---

### Overview

This version builds directly on V1 by stripping away the assumption 
that we KNOW the environment model. In V1, Value Iteration worked 
because we had perfect knowledge of P(s'|s,a) and R(s,a). But in 
real pursuit-evasion problems, you don't always get that luxury. 
The adversary changes behavior, the environment is noisy, and 
you have to LEARN from experience instead of planning from a model.

That's what V2 is about. Model-free RL i.e. learning what to do 
just by trying things and seeing what happens.

---

### What Confused Me Initially

Honestly, the hardest part was wrapping my head around the 
difference between on-policy and off-policy learning. In V1, 
there was just one policy and one value function. Now suddenly 
there are agents that learn about a DIFFERENT policy than the 
one they're actually following? That took a while to click, 
I don't know if that makes sense.

The second big confusion was eligibility traces i.e. how do you 
assign credit to decisions you made 5 steps ago that led to a 
reward NOW? TD(lambda) answers that question but understanding 
WHY it works took real drilling down.

---

### Algorithm Choice: Why DQN?

This was the most important decision for V2. The environment 
has discrete actions (UP, DOWN, LEFT, RIGHT), a manageable 
state space of 625 states, sparse rewards (only at goal/capture), 
and a stochastic adversary. I needed an algorithm that could 
handle all of that without unnecessary complexity.

Here is how I thought through the options:

| Algorithm | My Take | Verdict |
|-----------|---------|---------|
| **DQN** | Discrete actions map directly to output neurons. Experience replay breaks the correlation between sequential states in pursuit-evasion. Target network handles the stochastic adversary. Extends tabular Q-learning cleanly. | ✅ CHOSEN |
| REINFORCE | Too much variance with sparse rewards. No replay buffer means slow learning. | ❌ |
| Vanilla Actor-Critic | Lower variance than REINFORCE but overkill for 4 discrete actions. | ❌ |
| DDPG | Designed specifically for CONTINUOUS action spaces. Wrong fit entirely. | ❌ |
| TD3 | Same problem as DDPG - continuous actions only. | ❌ |
| PPO | State of the art and I respect it, but 625 states doesn't need that complexity. | ❌ |
| TRPO | Theoretically beautiful but the implementation overhead isn't justified here. | ❌ |
| SAC | Maximum entropy exploration is interesting, but again - continuous actions. | ❌ |

**Insight:** The pattern I noticed is that DDPG, TD3, and SAC are 
all built for continuous action spaces like robot joints or motor 
control. My environment is discrete. That eliminates half the list 
immediately. Between what's left, DQN is the most direct extension 
of the Q-learning I already implemented in tabular form, which 
means I can actually compare them as a proper ablation study.

**Why the cons are manageable:**
DQN can overestimate Q-values, but the target network addresses 
that directly. The state space is small enough that if DQN 
completely fails, tabular Q-learning is right there as a fallback. 
Hyperparameters are logged in configs/dqn.json so experiments 
are reproducible.

---

### Classical RL Algorithms Implemented

All algorithms below are in src/agents/ with matching config 
files in configs/.

| Algorithm | Type | Key Idea |
|-----------|------|----------|
| Q-Learning | Off-policy TD | Learns optimal policy regardless of exploration |
| SARSA | On-policy TD | Learns the policy it actually follows. Safer. |
| TD(λ) Backward | Eligibility traces | Credit flows backwards through visited states |
| TD(λ) Forward | Lambda-returns | Weighted average of all n-step returns |
| SARSA(λ) | On-policy traces | SARSA with eligibility trace credit assignment |
| SARSA(n) | n-step forward view | Bridges TD(0) and MC via hyperparameter n |
| Q(λ) | Off-policy traces | Q-learning with backwards credit propagation |
| Monte Carlo | Full episode returns | No bootstrapping. Unbiased but high variance. |
| TD(n) | n-step TD | n=1 is TD(0), n=infinity approaches MC |
| DQN | Deep off-policy TD | Neural network + replay buffer + target network |

---

### Pros and Cons of Model-Free RL (What I Actually Learned)

**Pros:**
No need to know P or R upfront - the agent figures it out by 
interacting with the environment. This is MUCH more realistic 
for real problems. Works even when the adversary changes 
behavior (stochastic transitions). Scales to problems where 
building a model is simply impossible.

**Cons:**
Sample efficiency is a REAL issue. Value iteration converged 
in 9 iterations in V1. Model-free methods need thousands of 
episodes. The exploration-exploitation tradeoff is now a 
design decision you have to make manually via epsilon. 
And debugging is harder because failures could be hyperparameters, 
not bugs in the code, I don't know if that makes sense.

---

### Tools Used

Python 3.13.2, NumPy, Matplotlib  
LLM: Claude AI for concept explanations, debugging, and implementation  
Collaborators: Course colleagues for conceptual discussions

---

### Academic References

- Sutton & Barto, "Reinforcement Learning: An Introduction", Chapters 5-7
- David Silver RL Course, Lectures 4-6
- Mnih et al. (2015), "Human-level control through deep reinforcement learning" (DQN paper)
- Course lecture materials (Week 5-6)

## License

MIT License - See LICENSE file for details

---

## Contact

**Uzezi Olorunmola**  
University of North Dakota  
Reinforcement Learning (Spring 2026)

---

*Last updated: February 11, 2026*
*Version 1 Submission*