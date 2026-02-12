# V1 Submission: Pursuit-Evasion GridWorld

Name: Uzezi Olorunmola  
Date: February 11, 2026  
Course: Reinforcement Learning (Spring 2026)

---

## Overview

This project implements a pursuit-evasion navigation task where an agent must reach a goal while avoiding a pursuing adversary. The big difference from basic GridWorld is that the state space includes BOTH the agent position AND the adversary position, which creates a 625-state MDP. Through this project, I learned about multi-agent dynamics, strategic reasoning, and how state space design affects problem complexity.

---

## Problem or Challenge

What confused me initially was understanding why we need to include the adversary position in the state. In basic GridWorld, the state is just (row, col). But here, the optimal action depends on WHERE THE ADVERSARY IS - if it's close, I need to avoid it; if it's far, I can move toward the goal. This means the state must be (agent_pos, adversary_pos), which explodes the state space from 25 states to 625 states.

Another really difficult challenge was handling collision states where the agent and adversary occupy the same cell. Initially I excluded these from the state space (thinking they're impossible), but that caused a KeyError during value iteration because transitions CAN lead to collisions. The solution was to include collision states and mark them as terminal with large negative rewards.

Also, understanding how the adversary patrol pattern affects the transition function took me a while to wrap my head around. The adversary moves deterministically in a clockwise pattern, so P(s'|s,a) depends on both the agent's action AND the adversary's scripted policy.

---

## Collaborators

I used Claude AI to help break down the state space expansion math, debug transition probability calculations, and structure the project repository. All code implementations were personally verified and understood.

---

## Key Learnings

What is 'Pursuit-Evasion MDP': Like navigation MDP but the environment includes a moving adversary

What is 'Joint State Space': State includes both agent position AND adversary position

What is 'Strategic Reasoning': Optimal action depends on adversary proximity, not just goal distance

What is 'Scripted Policy': Adversary follows deterministic patrol (V1), will learn in future versions

What is 'Non-Stationary Environment': Opponent behavior creates dynamic state transitions

These concepts built on the MDP foundations from earlier mini-projects, which helped a lot.

---

## MDP Formulation

### States (S)

States: (agent_position, adversary_position) where both positions ∈ {0,1,2,3,4}²

Size: 625 states (25 agent positions × 25 adversary positions)

Why this representation: The optimal action depends on WHERE the adversary is. If adversary is at (2,3) and agent is at (2,2), moving right would cause a collision. But if adversary is at (0,0), moving right is safe. This means we MUST include adversary position in the state.

Is it Markovian? Yes! The current (agent_pos, adv_pos) tells us everything we need to decide the next action. We don't need to remember where the adversary was 5 steps ago - the current position is enough because the adversary follows a deterministic patrol.

### Actions (A)

Actions: {UP, DOWN, LEFT, RIGHT}

Why 4 actions: Standard grid navigation. Agent doesn't control adversary (adversary moves automatically).

### Transition Function P(s'|s,a)

The transition has two components:

1. Agent movement: Deterministic based on action (with boundary checking)
2. Adversary movement: Scripted clockwise patrol around grid perimeter

Transition kernel:
```
s = (agent_pos, adv_pos)
agent_next = agent_pos + action (if valid, else stay)
adv_next = patrol_policy(adv_pos) (deterministic next position)
s' = (agent_next, adv_next)
```

Collision handling: If agent_next == adv_next, this is a terminal state with large negative reward.

### Reward Function R(s,a,s')
```
R(s, a, s') = 
    +10.0   if agent reaches goal (4,4)
    -10.0   if caught by adversary (agent_pos == adv_pos)
    -0.5    if adjacent to adversary (Manhattan distance ≤ 1)
    -0.1    otherwise (step cost)
```

Why these values:
- Goal reward (+10) and catch penalty (-10) are symmetric - success and failure equally important
- Proximity penalty (-0.5) encourages conservative play (stay away from adversary)
- Step cost (-0.1) promotes efficiency without overwhelming other rewards

Effect of rewards: With crash=-10, agent plays very safe and takes longer paths to avoid adversary. If we changed crash=-1, agent would take MORE risks and fly closer to adversary.

### Discount Factor γ

γ = 0.9

Why 0.9: Balances immediate safety (avoid adversary) vs long-term goal (reach target). Effective horizon ≈ 10 steps, which is reasonable for a 5×5 grid where optimal paths are 4-8 steps.

---

## Why Pursuit-Evasion? (Complexity Justification)

### Compared to Basic GridWorld

| Aspect | Basic GridWorld | Pursuit-Evasion |
|--------|----------------|-----------------|
| State space | 25 states | 625 states (25×) |
| Planning | Static pathfinding | Strategic avoidance |
| Difficulty | Trivial | Non-trivial |
| Real-world relevance | Low | High (robotics, games) |

### What Makes This Interesting

The key insight is that pursuit-evasion requires STRATEGIC REASONING, not just pathfinding. In basic GridWorld, the optimal action only depends on "where am I?" and "where is the goal?" But in pursuit-evasion, the optimal action also depends on "where is the adversary?" and "where will it be next?"

This is similar to:
- Healthcare optimization where resource allocation depends on patient flow (dynamic)
- Game AI where moves depend on opponent position (adversarial)
- Robotics where navigation must avoid moving obstacles (real-time)

Not just "find shortest path" - it's "find safest path given current threat".

---

## Algorithm: Value Iteration

### Implementation

I used tabular value iteration because the state space (625 states) is small enough to fit in memory. The algorithm is:
```
Initialize V(s) = 0 for all s
Repeat:
    For each state s = (agent_pos, adv_pos):
        For each action a in {UP, DOWN, LEFT, RIGHT}:
            Q(s,a) = R(s,a,s') + γ * V(s')
        V_new(s) = max_a Q(s,a)
    Δ = max_s |V_new(s) - V(s)|
Until Δ < θ (convergence threshold)

Extract policy: π(s) = argmax_a Q(s,a)
```

### Results

Convergence: 9 iterations (θ = 10⁻⁴)  
Computation time: ~10 seconds on standard laptop  
Policy quality: Agent successfully avoids adversary while reaching goal

Why so fast: Only 625 states, simple transitions, well-behaved rewards. Policy iteration would be even faster (typically 3-5 iterations) but value iteration is simpler to implement.

---

## Code Experiments

I ran experiments with different adversary starting positions to test robustness:

Adversary at (0,4) - top-right corner: Agent stays in left/center columns until adversary patrols away, then rushes to goal. Safe but slower.

Adversary at (2,2) - middle of grid: This was interesting! Agent has to navigate AROUND the adversary's patrol path. Policy shows more corrective moves and indirect routes.

Insight: The adversary starting position REALLY matters. When adversary starts near the goal, the agent has to wait for it to move away. When adversary starts far from goal, the agent can reach the goal quickly.

---

## What Worked

Breaking down the state space design first, THEN implementing transitions  
Running value iteration first, THEN trying to understand the learned policy  
Using visualization to see how agent avoids adversary  
Testing with different adversary positions to validate robustness

---

## What Didn't Work

Initially tried to exclude collision states from state space → caused KeyError  
First implementation had bug in adversary patrol (skipped corners) → fixed by explicitly listing patrol path  
Tried to visualize full 625-state value function → too complex, switched to summary plot

---

## Tools Used

Python 3.13, NumPy 2.2.5, Matplotlib 3.10.1  
Git/GitHub for version control  
Cookiecutter Data Science template for project structure  
Claude AI for concept explanations, debugging, and documentation review

---

## How to Run
```bash
cd rl-capstone-gridworld
pip install -r requirements.txt
python src/train.py
```

Outputs are saved to results/v1_baseline/

---

## Next Steps (V2 Roadmap)

### Immediate Improvements

Stochastic adversary: 80% follow patrol, 20% random move (tests robustness)  
Model-free learning: Q-learning and SARSA instead of value iteration  
Larger environment: 10×10 grid (10,000 states) to test scalability  
Better visualization: Animated GIF showing agent vs adversary over time

### Long-Term Vision (V3, Final)

Self-play: Both agent and adversary learn simultaneously  
Partial observability: Agent doesn't see adversary position (POMDP)  
Deep RL: DQN for even larger state spaces  
Real-world deployment: Transfer to actual robotics platform

Project scope may evolve based on course content and research interests.

---

## Formal MDP Components

States (S): {(i,j,k,l) | i,j,k,l ∈ [0,4]} ∪ {GOAL, CAUGHT}  
Actions (A): {UP, DOWN, LEFT, RIGHT}  
Initial Distribution (μ): δ((0,0), (0,4)) - agent at (0,0), adversary at (0,4)  
Transition Kernel P(s'|s,a): Defined by agent dynamics + adversary patrol  
Reward R(s,a,s'): As specified above  
Discount Factor (γ): 0.9  
Policy (π): Determined via value iteration  
Value Function V*(s): Expected cumulative reward following optimal policy

---

## Thoughts on the MDP Model

### Pros

MDPs give agent CONTROL over actions (unlike MRP where agent floats randomly)  
State space design is flexible - can include anything relevant to decision-making  
Value iteration guarantees optimal policy (if model is correct)  
Framework handles stochastic transitions naturally

### Cons

State space explosion is REAL - adding adversary increased states from 25 to 625  
Building transition model P(s'|s,a) manually is tedious and error-prone  
Assumes we know P and R perfectly (unrealistic for real-world)  
For continuous states would need function approximation (whole new complexity layer)

---

## References

Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press. Chapters 3-4 (MDP sections)

Littman, M. L. (1994). Markov games as a framework for multi-agent reinforcement learning. ICML.

David Silver RL Course, Lecture 3 (Planning by Dynamic Programming)

Gymnasium documentation: https://gymnasium.farama.org/

Course lecture materials (Week 2-5)

---

## Academic Integrity

Solo project: Uzezi Olorunmola

LLM assistance: Claude AI (Anthropic) for debugging, concept explanations, and documentation structure. Full conversation logs archived and available per course policy.

All code implementations personally verified and understood.

---

Last updated: February 11, 2026  
Version 1 Submission