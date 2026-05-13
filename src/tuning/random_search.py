import random

from src.tuning.search_space import SEARCH_SPACE


def sample_config(agent_name):

    space = SEARCH_SPACE[agent_name]

    return {
        key: random.choice(values)
        for key, values in space.items()
    }


def run_random_search(agent_name, n_trials=5):

    print(f"\nRunning random search for: {agent_name}")

    for i in range(n_trials):

        config = sample_config(agent_name)

        print(
            f"Trial {i+1}: {config}"
        )


if __name__ == "__main__":

    run_random_search(
        "qlearning"
    )