import random
from src.tuning.search_space import SEARCH_SPACE

def sample_config(agent_name):
    """
    Randomly sample hyperparameters
    from predefined search space
    """

    space = SEARCH_SPACE[agent_name]

    config = {
        key: random.choice(values)
        for key, values in space.items()
    }

    return config


if __name__ == "__main__":
    print(sample_config("qlearning"))
    print(sample_config("dqn"))