import numpy as np

def compute_metrics(returns):
    """
    Compute standard RL performance metrics
    """

    metrics = {
        "mean_return": np.mean(returns),
        "std_return": np.std(returns),
        "max_return": np.max(returns),
        "min_return": np.min(returns),
        "episodes": len(returns)
    }

    return metrics