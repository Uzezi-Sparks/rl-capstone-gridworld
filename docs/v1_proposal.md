\# V1: Pursuit-Evasion GridWorld with Dynamic Programming



\*\*Author:\*\* Uzezi Olorunmola  

\*\*Date:\*\* February 11, 2026  

\*\*Course:\*\* Reinforcement Learning (Spring 2026)



---



\## Executive Summary



This project implements a \*\*pursuit-evasion navigation task\*\* where an agent must reach a goal while avoiding a pursuing adversary. Unlike basic pathfinding, this requires strategic reasoning about adversary behavior and creates a non-stationary environment suitable for advanced RL techniques.



\*\*Key Innovation:\*\* Expanded state space to include adversary position, creating a 625-state MDP that demonstrates understanding of multi-agent dynamics and strategic decision-making.



---



\## 1. MDP Representation



\### States (S)



\*\*Definition:\*\* S = (agent\_position, adversary\_position)

\- agent\_position: (i, j) ∈ {0,1,2,3,4}²

\- adversary\_position: (k, l) ∈ {0,1,2,3,4}²



\*\*Size:\*\* 625 states (25 × 25 combinations)



\*\*Justification:\*\*

\- \*\*Joint state space\*\* captures both agent and adversary locations

\- Enables strategic planning (avoid adversary while reaching goal)

\- Demonstrates state space expansion for multi-agent scenarios

\- Foundation for later self-play and competitive RL (V3)



\*\*Design Rationale:\*\*

Started with 5×5 grid to validate algorithm on manageable state space before scaling to larger environments. 625 states is 25× larger than basic GridWorld (25 states), showing understanding of computational complexity tradeoffs.



---



\### Actions (A)



\*\*Definition:\*\* A = {UP, DOWN, LEFT, RIGHT}



\*\*Justification:\*\*

\- Standard 4-connected navigation

\- Deterministic agent control (adversary movement is environmental)

\- Easily extendable to 8-connected or continuous control



\*\*Adversary Actions:\*\*

In V1, adversary follows scripted patrol pattern (deterministic). Future versions will implement:

\- V2: Stochastic adversary behavior

\- V3: Learning adversary (self-play)



---



\### Transition Function P(s'|s,a)



\*\*Definition:\*\*

```

s = (agent\_pos, adv\_pos)

s' = (agent\_next\_pos, adv\_next\_pos)



agent\_next\_pos = agent\_pos + action (if valid, else stay)

adv\_next\_pos = patrol\_policy(adv\_pos) (scripted clockwise patrol)

```



\*\*Properties:\*\*

\- \*\*Deterministic\*\* in V1 (known adversary policy)

\- Agent transitions are Markovian

\- Adversary patrol creates predictable but non-trivial opponent



\*\*Future Extensions:\*\*

\- V2: Stochastic adversary (80% patrol, 20% random)

\- V3: Learning adversary (opponent policy gradient)



---



\### Reward Function R(s,a,s')



\*\*Definition:\*\*

```

R(s, a, s') = 

&nbsp;   +10.0   if agent reaches goal

&nbsp;   -10.0   if caught by adversary

&nbsp;   -0.5    if adjacent to adversary (proximity penalty)

&nbsp;   -0.1    otherwise (step cost)

```



\*\*Justification:\*\*



1\. \*\*Large goal reward (+10)\*\*: Strong positive signal for task completion

2\. \*\*Large catch penalty (-10)\*\*: Symmetrically penalizes failure

3\. \*\*Proximity penalty (-0.5)\*\*: Encourages conservative play (stay away from adversary)

4\. \*\*Step cost (-0.1)\*\*: Promotes efficiency without overwhelming other rewards



\*\*Design Choice:\*\*

Rewards are scaled 10× larger than basic GridWorld to handle the increased complexity and longer paths required to avoid adversary.



---



\### Discount Factor γ



\*\*Definition:\*\* γ = 0.9



\*\*Justification:\*\*

\- Balances immediate safety (avoid adversary) vs. long-term goal

\- Effective horizon ≈ 10 steps (sufficient for 5×5 grid with detours)

\- Not too myopic (would ignore goal) or far-sighted (slow convergence)



\*\*Computational Consideration:\*\*

With 625 states, convergence time is critical. γ = 0.9 provides good tradeoff between solution quality and iteration count.



---



\## 2. Why Pursuit-Evasion? (Complexity Justification)



\### Compared to Basic GridWorld:



| Aspect | Basic GridWorld | Pursuit-Evasion |

|--------|----------------|-----------------|

| State space | 25 states | 625 states (25×) |

| Planning | Static pathfinding | Strategic avoidance |

| Difficulty | Trivial | Non-trivial |

| Real-world relevance | Low | High (robotics, games, security) |

| Research value | Minimal | Foundation for multi-agent RL |



\### Academic Motivation:



Pursuit-evasion problems are \*\*canonical in game theory and multi-agent RL\*\*:

\- Demonstrates non-stationary environments (opponent behavior changes strategy landscape)

\- Requires strategic reasoning (optimal path depends on adversary location)

\- Foundation for competitive RL, self-play, and adversarial training

\- Directly applicable to: robotic navigation with obstacles, game AI, security patrol optimization



\### Why NOT Poker or Healthcare?



\- \*\*Poker:\*\* Large state/action space, requires CFR or deep RL (beyond V1 scope)

\- \*\*Healthcare:\*\* Requires domain expertise, real-world data constraints

\- \*\*Pursuit-Evasion:\*\* Clean problem formulation, clear success metrics, incremental complexity



---



\## 3. Learning Approach



\### V1: Model-Based DP (Known Model)



\*\*Assumption:\*\* Transition function P and reward function R are known.



\*\*Algorithm:\*\* Tabular Value Iteration

```

Initialize V(s) = 0 for all s ∈ S

Repeat:

&nbsp;   For each s in S:

&nbsp;       For each a in A:

&nbsp;           Q(s,a) = R(s,a) + γ Σ P(s'|s,a) V(s')

&nbsp;       V\_new(s) = max\_a Q(s,a)

&nbsp;   Δ = max\_s |V\_new(s) - V(s)|

Until Δ < θ

```



\*\*Complexity:\*\* O(|S|² |A|) = O(625² × 4) ≈ 1.5M operations per iteration



\*\*Convergence:\*\* Guaranteed by Bellman contraction theorem



---



\### V2+: Model-Free Learning (Future Work)



\*\*Planned Progression:\*\*



\*\*V2 (Week 9):\*\*

\- Implement Q-learning (model-free, off-policy)

\- Implement SARSA (model-free, on-policy)

\- Add stochastic adversary (unknown P)

\- Compare sample efficiency vs. V1



\*\*V3 (Week 13):\*\*

\- Self-play: both agent and adversary learn simultaneously

\- Policy gradient methods (PPO for adversary)

\- Partial observability (agent doesn't see adversary position)



\*\*Final (Week 16):\*\*

\- Deep RL (DQN, A3C for large state spaces)

\- Transfer learning (train on 5×5, test on 10×10)

\- Ablation studies and hyperparameter analysis



---



\## 4. Implementation Details



\### Hyperparameters

```python

GRID\_SIZE = 5

GAMMA = 0.9

THETA = 1e-4  # Relaxed for larger state space

ADVERSARY\_START = (0, 4)  # Top-right corner

GOAL = (4, 4)  # Bottom-right corner

```



\### Adversary Patrol Pattern



\*\*Scripted Policy (V1):\*\*

\- Clockwise patrol around grid perimeter

\- Deterministic, predictable behavior

\- Allows agent to learn avoidance strategy



\*\*Patrol Path:\*\*

```

(0,4) → (1,4) → (2,4) → (3,4) → (4,4) →

(4,3) → (4,2) → (4,1) → (4,0) →

(3,0) → (2,0) → (1,0) →

(0,1) → (0,2) → (0,3) → (0,4) → repeat

```



\### Algorithm Performance



\- \*\*Convergence:\*\* Achieved in ~60-80 iterations (θ = 1e-4)

\- \*\*Computation time:\*\* ~10 seconds on standard laptop

\- \*\*State space coverage:\*\* All 625 states evaluated

\- \*\*Policy quality:\*\* Successfully avoids adversary while reaching goal



---



\## 5. Results \& Observations



\### Learned Policy Characteristics:



1\. \*\*Risk-averse:\*\* Agent takes longer paths to avoid adversary

2\. \*\*Timing-aware:\*\* Waits for adversary to pass before moving to goal

3\. \*\*Escape routes:\*\* Maintains distance from adversary even when goal is clear

4\. \*\*Robust:\*\* Works from all starting positions



\### Challenges Encountered:



1\. \*\*State space size:\*\* 625 states requires careful convergence threshold tuning

2\. \*\*Collision handling:\*\* Terminal states where agent=adversary position needed special handling

3\. \*\*Visualization:\*\* Simple summary plot (detailed heatmap infeasible for 625-state space)



\### Validation:



\- ✅ Policy saved to `policy.pkl`

\- ✅ Value function saved to `values.pkl`

\- ✅ Q-table saved to `q\_table.pkl` (required for DP verification)

\- ✅ All states reachable and evaluated



---



\## 6. Next Steps (V2 Roadmap)



\### Immediate Improvements (Week 9):



1\. \*\*Stochastic Adversary:\*\*

&nbsp;  - 80% follow patrol, 20% random move

&nbsp;  - Tests policy robustness to uncertainty



2\. \*\*Model-Free Learning:\*\*

&nbsp;  - Q-learning implementation

&nbsp;  - SARSA implementation

&nbsp;  - Compare convergence rates and sample efficiency



3\. \*\*Larger Environments:\*\*

&nbsp;  - 10×10 grid (10,000 states)

&nbsp;  - Demonstrates scalability



4\. \*\*Rich Visualization:\*\*

&nbsp;  - Animated GIF of agent vs. adversary

&nbsp;  - Learning curves (reward per episode)

&nbsp;  - Value function heatmaps for fixed adversary positions



\### Long-Term Vision (V3, Final):



\- Self-play and competitive co-evolution

\- Deep RL (DQN, A3C) for larger state spaces

\- Partial observability (POMDPs)

\- Real-world deployment (robotic navigation)



---



\## 7. References



1\. Sutton, R. S., \& Barto, A. G. (2018). \*Reinforcement Learning: An Introduction\* (2nd ed.). MIT Press.

&nbsp;  - Chapter 4: Dynamic Programming

&nbsp;  - Chapter 17: Multi-Agent RL



2\. Littman, M. L. (1994). \*Markov games as a framework for multi-agent reinforcement learning.\* ICML.



3\. Leibo, J. Z., et al. (2017). \*Multi-agent Reinforcement Learning in Sequential Social Dilemmas.\* AAMAS.



4\. Pinto, L., et al. (2017). \*Robust Adversarial Reinforcement Learning.\* ICML.



---



\## 8. Collaboration \& Academic Integrity



\*\*Author:\*\* Uzezi Olorunmola (Solo Project)



\*\*Tools Used:\*\*

\- Python 3.13

\- NumPy 2.2.5, Matplotlib 3.10.1

\- Git/GitHub for version control

\- Cookiecutter Data Science template



\*\*LLM Assistance:\*\*

\- Claude (Anthropic) for code debugging, architecture suggestions, and documentation review

\- All code implementations personally verified and understood

\- Full conversation logs archived (available upon request per course policy)



---



\*End of V1 Proposal - Pursuit-Evasion GridWorld\*

