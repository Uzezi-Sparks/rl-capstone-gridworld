# Reinforcement Learning Capstone: Pursuit–Evasion GridWorld (V1 → V3)

## Project Overview


This project was started with a simple question in mind: what happens when an environment gradually becomes less predictable?

Rather than jumping straight into complex Reinforcement Learning algorithms, I approached the problem as a progression. Each version had a job to do. Every stage introduced a little more uncertainty, a little more interaction, and a little more realism than the one before it.



V1 established the foundation. The environment was controlled and fully known. This made it possible to apply Dynamic Programming methods such as Value Iteration and build a baseline for optimal decision-making under certainty.



V2 shifted things in a different direction. Instead of assuming the environment was well known or understood, the agent now had to learn through interaction. This introduced model-free Reinforcement Learning methods including Q-Learning, SARSA, TD variants and Deep Q-Networks (DQN). The environment itself also became more dynamic. A pursuit–evasion setting was introduced where an agent attempted to reach a goal while avoiding an adversary with partially stochastic behavior.



Now this is where things got interesting.

As uncertainty increased, performance also became less predictable. Exploration strategies, adversary movement, and environmental randomness meant that repeated runs did not always produce identical outcomes. That was expected. More importantly, it raised a new question.



How do we know whether an observed result is genuinely better, or simply a good run?



That question became the motivation behind V3.



V3 does not replace earlier versions. It builds on them. The focus shifted from simply training agents toward optimizing, testing, and validating them through structured experimentation. Hyperparameter search, experiment tracking, repeated run testing, and performance comparison were introduced to explore behaviors under uncertainty instead of single run results. What the goal had become by the end of the project was no longer to train an agent to achieve a goal.



The real issue was the broader issue of what happens when RL systems get more complex, and how systematic evaluation often means better decision making under uncertainty.

---



## Project Evolution


### V1 — Building the Foundation



The first version focused on establishing a controlled baseline. The environment dynamics were fully known, making it possible to apply Dynamic Programming techniques and compute policies directly.



Key focus areas:



- Known environment transitions
- Value Iteration baseline
- Policy computation under certainty
- Establishing a performance reference point


This stage answered an important question:



*Can an optimal policy be computed when the environment is fully understood?*



---



### V2 — Learning Through Interaction



V2 moved away from certainty and introduced interaction.



Rather than giving the agent complete knowledge of the environment, the agent had to learn behavior through experience. The GridWorld evolved into a pursuit–evasion environment where reaching a goal was no longer enough. The agent also had to avoid an adversary operating under partially stochastic behavior.



Several Reinforcement Learning methods were introduced:



- Q-Learning
- SARSA
- TD(λ)
- Monte Carlo methods
- Deep Q-Networks (DQN)



This stage introduced:



- exploration using epsilon-greedy policies
- stochastic adversary movement
- larger state representation
- uncertainty in outcomes



The project moved from:



```text
solve known dynamics
      ↓
learn through interaction
```

---

### V3 — Optimize, Test and Validate



As uncertainty increased, repeated runs began producing different outcomes.



This was not treated as a problem. It became part of the story.



Variability introduced a new challenge: evaluating performance could no longer depend on isolated runs. A stronger process was needed.



V3 introduced:



- Random hyperparameter search
- Experiment logging
- Performance metrics
- Repeated trials
- Visualization and evidence tracking


The goal shifted toward:

```text
train
   ↓
evaluate
   ↓
compare
   ↓
optimize
```

---



## Environment Design



The project uses a custom Pursuit–Evasion GridWorld environment built around a simple idea: reaching a goal should not automatically mean the task is easy.



The environment consists of:



- A 5×5 GridWorld
- An agent attempting to reach a goal state
- An adversary following a partially stochastic movement policy
- Reward and penalty structures encouraging efficient behavior


The environment was intentionally designed to evolve alongside the project to be as close to reality as possible.



Early versions emphasized control and predictability. Later versions introduced uncertainty through exploration policies and stochastic adversary behavior. This made learning less straightforward, but also more representative of real-world decision-making challenges.



### Reward Structure



| Event | Reward |

|---|---:|

| Reach goal | +10 |

| Caught by adversary | -10 |

| Step cost | -0.1 |

| Near adversary | -0.5 |



The reward structure created an important tradeoff:



The shortest path was not always the safest path.



Agents therefore had to balance exploration, efficiency and risk.


## Learning Strategies



Alongside the environmental developments, learning strategies also evolved.



Not just new algorithms were introduced for each version – they helped understand how learning behavior would change depending on assumptions about the environment and its complexity.



### Dynamic Programming-Based Strategies (V1)



Dynamic programming-based strategies, used in V1, assumed full information about environment modeling.



These included:



- Value iteration algorithm
- Policy evaluation principles
- Computation of optimal policy



They operated well in deterministic environment because transitions were known in advance.



---




### Reinforcement Learning Strategies (V2)



For V2, model-free learning methods that used agent-environment interactions to learn instead of complete information about the environment were considered.



Included:



- Q-Learning

- SARSA

- Temporal Difference methods

- Monte Carlo methods

- Deep Q-Networks (DQN)



The whole scenario changed dramatically.



The agent did not solve the environment anymore, but had to explore, make errors, gain experience and become better through interaction.



---







### Optimization and Evaluation Techniques (V3)




When it came to V3, training itself was no longer the only objective.



Uncertainty of repeated experiments revealed the randomness of their results. It became important to have other methods as well:



- Random hyperparameter optimization
- Experiment logging
- Performance measurement
- Iterative evaluation runs
- Visualization of results

This transformed the workflow from:



```text
train once
     ↓
observe result
```



## Key Findings



Among the most intriguing findings of this particular experiment is how variance itself was a key result worth considering.



Once policies for exploration were implemented and stochastic movements of the adversary were added alongside an increased state space representation, variability of results upon repeated experiments was inevitable.



This was expected.



It was not whether individual runs themselves were different, but whether certain regularities would still be discernible within their variance.



For example, some of the discovered hyperparameters combinations included:



| Alpha | Gamma | Epsilon | Mean Return |

|---:|---:|---:|---:|

| 0.2 | 0.95 | 0.10 | -7.46 |

| 0.2 | 0.95 | 0.05 | -9.46 |

| 0.1 | 0.95 | 0.05 | -10.10 |

| 0.2 | 0.95 | 0.20 | -9.40 |



Some regularities started emerging.



Specifically, higher gamma values appeared in all successful configurations. This suggests that prioritizing future reward might have positively impacted results for the pursuit-evasion scenario.



However, none of the configurations proved optimal in each and every test case.



This finding is important since, rather than leading towards a conclusion regarding a single optimal hyperparameter setting, this result reiterates the necessity of running multiple tests instead of just one or two runs.

---

## Exploratory Extension: Partial Observability

As a final extension, V3 moved beyond optimization alone and explored an uncertainty-inspired environment modification motivated by POMDP concepts discussed in class.

Earlier versions assumed complete state information. The agent always knew where the adversary was located.

To gradually move toward more realistic environments, a visibility mechanism was introduced.

Key additions:

- Visibility radius parameter
- Hidden adversary states outside local range
- Observation filtering
- Visibility-based evaluation experiments

The environment behavior became:

```text
Full observability
       ↓
limited visibility
       ↓
partial information
```

Rather than always observing the adversary position directly, the agent only received information when the adversary was within a specified visibility radius.

Example observations:

Radius 1 → 33% visible  
Radius 2 → 67% visible  
Radius 5 → 100% visible  

This extension was intentionally exploratory rather than a full POMDP implementation. The objective was to investigate how reducing information availability changes environmental uncertainty and creates a foundation for future partially observable RL research.

---

## Conclusion



In summary, this project turned out to be not only an exercise in applying Reinforcement Learning algorithms.



Where initially there were relatively clear goals, at some point, this developed into something bigger – the study of how learning systems interact with increasing uncertainty.



The step from version 1, through 2, and finally to 3 proved to be as important as the actual application of algorithms.



Version 1 gave structure.



Version 2 brought interactivity and randomness.



And the final step allowed for a testing, tuning and validating phase, which helped make the results more tangible.



One realization became increasingly clear throughout all stages of the project:



While in artificial worlds performance metrics were quite straightforward, as environments get increasingly realistic evaluation plays just as big a role as training.



It was never the goal to develop an agent capable of reaching a goal state.



Rather, it was about seeing how systems would behave under increased complexity, randomness, and unpredictable situations.

---

## Project Structure



```text

rl-capstone-gridworld/

│

├── configs/

├── reports/

│   ├── evidence/

│   └── figures/

│

├── src/

│   ├── agents/

│   ├── environment/

│   ├── models/

│   ├── replay_buffer/

│   ├── evaluation/

│   ├── tuning/

│   ├── utils/

│   └── train_v3.py

│

└── README_v3.md

```

---

## Academic Integrity

**Solo Project:** Uzezi Olorunmola

**Development Tools:**
- Python 3.13, NumPy 2.2.5, Matplotlib 3.10.1
- Git/GitHub, Cookiecutter Data Science template
- ChatGpt & Claude AI - debugging assistance and documentation review
- Full LLM conversation logs archived (available per course policy)

All code implementations personally verified and understood.

---


