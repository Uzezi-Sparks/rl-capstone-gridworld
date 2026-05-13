import json
import os
from datetime import datetime

def save_experiment(metrics, algorithm, save_dir="results/experiments"):
    
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filepath = os.path.join(
        save_dir,
        f"{algorithm}_{timestamp}.json"
    )

    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved experiment: {filepath}")