from src.environment.gridworld import PursuitEvasionEnv

from src.evaluation.metrics import compute_metrics


class PursuitEvasionEnvV3(PursuitEvasionEnv):

    def __init__(
        self,
        grid_size=5,
        goal=(4,4),
        adversary_start=(0,4),
        stochastic_prob=0.25,
        visibility_radius=2
    ):

        super().__init__(
            grid_size=grid_size,
            goal=goal,
            adversary_start=adversary_start,
            stochastic_prob=stochastic_prob
        )

        # V3 exploratory extension:
        # partial observability inspired setting
        self.visibility_radius = visibility_radius


    def get_observation(self, agent_pos, adv_pos):
    
        """
        Partial observability:
        agent sees adversary only if nearby
        """

        distance = (
            abs(agent_pos[0] - adv_pos[0]) +
            abs(agent_pos[1] - adv_pos[1])
        )

        if distance <= self.visibility_radius:
            observed_adv = adv_pos

        else:
            observed_adv = ("UNKNOWN", "UNKNOWN")

        return (agent_pos, observed_adv)

   

    def observation_summary(self, agent_pos, adv_pos):
        observation =self.get_observation(
            agent_pos,
            adv_pos
        )
        if observation[1] == ("UNKNOWN","UNKNOWN"):
            return "Adversary not visible"
        return f"Adversary detected at {observation[1]}"

    def visibility_status(self, agent_pos, adv_pos):

        distance = (
            abs(agent_pos[0]-adv_pos[0]) +
            abs(agent_pos[1]-adv_pos[1])
        )

        return {
            "distance": distance,
            "visible": distance <= self.visibility_radius
        }

    def evaluate_visibility(self, scenarios, agent_pos):

        returns = []

        for _, adv in scenarios:

            status = self.visibility_status(
                agent_pos,
                adv
            )

            if status["visible"]:
                returns.append(1)

            else:
                returns.append(0)

        return returns

if __name__ == "__main__":

    env = PursuitEvasionEnvV3()

    agent = (2,2)

    scenarios = [
        ("Close", (3,2)),
        ("Medium", (4,2)),
        ("Far", (0,4))
    ]

    for label, adv in scenarios:

        print(f"\n{label} adversary:")

        print(
            env.get_observation(
                agent,
                adv
            )
        )

        print(
            env.observation_summary(
                agent,
                adv
            )
        )
        
        print(
            env.visibility_status(
                agent,
                adv
            )
        )


    print("\n--- Visibility Evaluation Summary ---")

    visible_count = 0

    for _, adv in scenarios:

        status = env.visibility_status(
            agent,
            adv
        )

        if status["visible"]:
            visible_count += 1

    print(
        f"Visible scenarios: "
        f"{visible_count}/{len(scenarios)}"
    )

    visibility_returns = env.evaluate_visibility(
        scenarios,
        agent
    )

    metrics = compute_metrics(
        visibility_returns
    )

    print("\n--- Visibility Metrics ---")

    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    print("\n--- Visibility Radius Comparison ---")

    radii = [1,2,5]

    for r in radii:

        env.visibility_radius = r

        visible = 0

        for _, adv in scenarios:

            status = env.visibility_status(
                agent,
                adv
            )

            if status["visible"]:
                visible += 1

        ratio = visible/len(scenarios)

        print(
            f"Radius {r}: "
            f"{visible}/{len(scenarios)} visible "
            f"({ratio:.2f})"
        )