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
    Temporary V3 placeholder.

    Simulates experiment returns.
    Real training loop comes later.
    """

    returns = [1, 2, 4, 3, 5]

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