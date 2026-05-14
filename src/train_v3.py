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

    config = load_config(
        args.agent
    )

    env = PursuitEvasionEnv(
        stochastic_prob=0.25
    )

    agent = build_agent(
        args.agent,
        config
    )

    print(f"Loaded: {args.agent}")
    print("V3 orchestration initialized")

    returns = []

    for episode in range(5):

        state = (
            (0, 0),
            env.adversary_start
        )

        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 50:

            agent_pos, adv_pos = state
	    
            state_idx = (
		(agent_pos[0]*env.size + agent_pos[1])
		*(env.size * env.size)
		+ (adv_pos[0] * env.size + adv_pos[1])
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

            if hasattr(agent, "update"):
		
                next_agent, next_adv = next_state
                next_idx = (
                    (next_agent[0] * env.size + next_agent[1])
                    * (env.size * env.size)
                    + (next_adv[0] * env.size + next_adv[1])  

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

    metrics = compute_metrics(
        returns
    )

    save_experiment(
        metrics,
        args.agent
    )

    print(metrics)