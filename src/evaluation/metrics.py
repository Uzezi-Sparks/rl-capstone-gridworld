import numpy as np

def compute_metrics(returns):
    """
    Compute standard RL performance metrics
    """

    metrics = {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "max_return": float(np.max(returns)),
        "min_return": float(np.min(returns)),
        "episodes": int(len(returns))
    }

    return metrics