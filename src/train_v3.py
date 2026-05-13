import argparse
import json

from src.environment.gridworld import PursuitEvasionEnv
from src.agents.qlearning import QLearningAgent
from src.agents.dqn import DQNAgent

from src.evaluation.metrics import compute_metrics
from src.evaluation.experiment_logger import save_experiment


def load_config(agent_name):
    with open(f"configs/{agent_name}.json", "r") as f:
        return json.load(f)


def build_agent(agent_name, config):

    hp = config["hyperparameters"]

    if agent_name == "qlearning":
        return QLearningAgent(
            n_states=625,
            n_actions=4,
            alpha=hp["alpha"],
            gamma=hp["gamma"],
            epsilon=hp["epsilon"]
        )

    elif agent_name == "dqn":
        return DQNAgent(
            state_size=4,
            n_actions=4,
            alpha=hp["alpha"],
            gamma=hp["gamma"],
            epsilon=hp["epsilon_start"],
            epsilon_min=hp["epsilon_min"],
            epsilon_decay=hp["epsilon_decay"],
            batch_size=hp["batch_size"],
            buffer_size=hp["buffer_size"],
            target_update_freq=hp["target_update_freq"]
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--agent",
        type=str,
        required=True
    )

    args = parser.parse_args()

    config = load_config(args.agent)

    env = PursuitEvasionEnv(
        stochastic_prob=0.25
    )

    agent = build_agent(
        args.agent,
        config
    )

    print(f"Loaded: {args.agent}")
    print("V3 orchestration initialized")

    dummy_returns = [1, 2, 3, 4, 5]

    metrics = compute_metrics(dummy_returns)

    save_experiment(
        metrics,
        args.agent
    )

    print(metrics)