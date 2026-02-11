"""
V1 Training Script - GridWorld Value Iteration
Runs DP algorithm and saves all required artifacts
"""

import sys
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from environment.gridworld import PursuitEvasionEnv
from agents.value_iteration import ValueIteration

def visualize(env, V, policy, Q, save_path):
    """Create simple visualization for pursuit-evasion"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Show summary info
    num_states = len(V)
    ax.text(0.5, 0.7, 'Pursuit-Evasion GridWorld', 
            ha='center', fontsize=20, fontweight='bold')
    ax.text(0.5, 0.55, f'State Space: {num_states} states', 
            ha='center', fontsize=16)
    ax.text(0.5, 0.45, f'Grid Size: {env.size}×{env.size}', 
            ha='center', fontsize=14)
    ax.text(0.5, 0.35, f'Agent Goal: {env.goal}', 
            ha='center', fontsize=14)
    ax.text(0.5, 0.25, f'Adversary Start: {env.adversary_start}', 
            ha='center', fontsize=14)
    ax.text(0.5, 0.1, '✓ Policy Computed Successfully', 
            ha='center', fontsize=12, color='green', fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

def main():
    print("="*60)
    print("V1: GRIDWORLD VALUE ITERATION")
    print("="*60)
    
# Hyperparameters
GRID_SIZE = 5
GAMMA = 0.9
THETA = 1e-4  # Relaxed for larger state space
ADVERSARY_START = (0, 4)  # Top-right corner

print(f"\nHyperparameters:")
print(f"  Grid: {GRID_SIZE}x{GRID_SIZE}")
print(f"  γ (discount): {GAMMA}")
print(f"  θ (threshold): {THETA}")
print(f"  Adversary start: {ADVERSARY_START}\n")

# Create environment
env = PursuitEvasionEnv(grid_size=GRID_SIZE, goal_state=(4,4), adversary_start=ADVERSARY_START)

print(f"Environment: Pursuit-Evasion GridWorld")
print(f"Agent goal: {env.goal}")
print(f"State space size: {len(env.get_all_states())} states\n")

# Run value iteration  ← Should align with the prints above
vi = ValueIteration(env, gamma=GAMMA, theta=THETA)
V = vi.run()
    
  
# Extract policy and Q-table
policy = vi.extract_policy()
Q = vi.get_q_table()

# Create results directory
os.makedirs('results/v1_baseline', exist_ok=True)

# Save artifacts
with open('results/v1_baseline/values.pkl', 'wb') as f:
    pickle.dump(V, f)
print("\n✓ Saved: results/v1_baseline/values.pkl")

with open('results/v1_baseline/policy.pkl', 'wb') as f:
    pickle.dump(policy, f)
print("✓ Saved: results/v1_baseline/policy.pkl")

with open('results/v1_baseline/q_table.pkl', 'wb') as f:
    pickle.dump(Q, f)
print("✓ Saved: results/v1_baseline/q_table.pkl")

# Visualize
visualize(env, V, policy, Q, 'results/v1_baseline/gridworld.png')

print("\n" + "="*60)
print("✅ V1 SUBMISSION READY")
print("="*60)
print("\nNext: Update README.md and docs/v1_proposal.md")

if __name__ == "__main__":
    main()