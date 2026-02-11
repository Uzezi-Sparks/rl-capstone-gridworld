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

### 🔄 V2 (Week 9) - Model-Free Learning
- [ ] Stochastic adversary behavior
- [ ] Q-learning implementation
- [ ] SARSA implementation
- [ ] Learning curves & ablations

### 🔄 V3 (Week 13) - Self-Play
- [ ] Learning adversary (policy gradient)
- [ ] Partial observability (POMDP)
- [ ] 10×10 environment (10K states)
- [ ] Animated visualizations

### 🔄 Final (Week 16) - Deep RL
- [ ] DQN for large state spaces
- [ ] Multi-agent deep RL (A3C)
- [ ] Transfer learning experiments
- [ ] Research-quality writeup

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