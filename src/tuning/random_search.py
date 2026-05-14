import random

from src.tuning.search_space import SEARCH_SPACE
from src.evaluation.metrics import compute_metrics


def sample_config(agent_name):

    space = SEARCH_SPACE[agent_name]

    return {
        key: random.choice(values)
        for key, values in space.items()
    }



def mock_experiment(config):
    """
    Run lightweight V3 experiment
    using actual environment interaction
    """

    from src.environment.gridworld import PursuitEvasionEnv
    from src.agents.qlearning import QLearningAgent

    env = PursuitEvasionEnv(
        stochastic_prob=0.25
    )

    agent = QLearningAgent(
        n_states=625,
        n_actions=4,
        alpha=config["alpha"],
        gamma=config["gamma"],
        epsilon=config["epsilon"]
    )

    returns = []

    for episode in range(5):

        state = (
            (0,0),
            env.adversary_start
        )

        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 50:

            agent_pos, adv_pos = state

            state_idx = (
                (agent_pos[0]*env.size+agent_pos[1])
                *(env.size*env.size)
                +(adv_pos[0]*env.size+adv_pos[1])
            )

            action = agent.get_action(
                state_idx
            )

            next_state = env.get_next_state(
                state,
                action
            )

            reward = env.get_reward(
                state,
                action,
                next_state
            )

            done = env.is_terminal(
                next_state
            )

            next_agent, next_adv = next_state

            next_idx = (
                (next_agent[0]*env.size+next_agent[1])
                *(env.size*env.size)
                +(next_adv[0]*env.size+next_adv[1])
            )

            agent.update(
                state_idx,
                action,
                reward,
                next_idx,
                done
            )

            total_reward += reward
            state = next_state
            steps += 1

        returns.append(
            total_reward
        )

    return compute_metrics(
        returns
    )
    """
    Lightweight pseudo-training run.

    Mimics reward variation based on sampled
    hyperparameters so search behaves realistically.
    """

    alpha = config["alpha"]
    gamma = config["gamma"]
    epsilon = config["epsilon"]

    score = (
        gamma * 10
        + alpha * 5
        - epsilon * 2
    )

    returns = [
        score - 1,
        score,
        score + 1,
        score - 0.5,
        score + 0.5
    ]

    return compute_metrics(
        returns
    )



def run_random_search(agent_name, n_trials=5):

    print(f"\nRunning random search: {agent_name}")

    best_score = -9999
    best_config = None

    for i in range(n_trials):

        config = sample_config(
            agent_name
        )

        metrics = mock_experiment(
            config
        )

        score = metrics["mean_return"]

        print(
            f"\nTrial {i+1}"
        )

        print(config)

        print(metrics)

        if score > best_score:

            best_score = score
            best_config = config

    print("\nBest configuration:")

    print(best_config)

    print(
        f"Best score: {best_score}"
    )


if __name__ == "__main__":

    run_random_search(
        "qlearning"
    )