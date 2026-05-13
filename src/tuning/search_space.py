SEARCH_SPACE = {
    "qlearning": {
        "alpha": [0.01, 0.05, 0.1, 0.2],
        "gamma": [0.8, 0.9, 0.95],
        "epsilon": [0.05, 0.1, 0.2]
    },

    "dqn": {
        "alpha": [0.0001, 0.001, 0.005],
        "gamma": [0.8, 0.9, 0.95],
        "batch_size": [16, 32, 64],
        "epsilon_decay": [0.99, 0.995]
    }
}