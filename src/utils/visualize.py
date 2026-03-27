import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

"""
Visualization Utilities for PursuitEvasionEnv

Includes:
- Policy heatmap: what action does the agent take in each state?
- Q-value heatmap: how valuable is each state?
- Saliency map: which input features matter most to the DQN?
- Learning curve plotter
"""

ACTION_SYMBOLS = {0: '↑', 1: '↓', 2: '←', 3: '→'}
ACTION_NAMES   = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}

def plot_policy_heatmap(Q_table, grid_size=5, adv_pos=(0,4), 
                        save_path='results/policy_heatmap.png'):
    """
    Show greedy policy on the grid for a fixed adversary position.
    Each cell shows the best action the agent would take.
    """
    policy_grid = np.zeros((grid_size, grid_size), dtype=int)
    value_grid  = np.zeros((grid_size, grid_size))

    for row in range(grid_size):
        for col in range(grid_size):
            # Encode state as flat index
            agent_pos = (row, col)
            state_idx = (agent_pos[0] * grid_size + agent_pos[1]) * \
                        (grid_size * grid_size) + \
                        (adv_pos[0] * grid_size + adv_pos[1])
            if state_idx < len(Q_table):
                best_action = np.argmax(Q_table[state_idx])
                policy_grid[row, col] = best_action
                value_grid[row, col]  = np.max(Q_table[state_idx])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Policy & Value Map (Adversary at {adv_pos})', 
                 fontsize=14, fontweight='bold')

    # Left: Policy arrows
    ax = axes[0]
    ax.set_title('Greedy Policy', fontsize=12)
    im = ax.imshow(value_grid, cmap='Blues', alpha=0.4)
    for row in range(grid_size):
        for col in range(grid_size):
            symbol = ACTION_SYMBOLS[policy_grid[row, col]]
            ax.text(col, row, symbol, ha='center', va='center', fontsize=20)
    # Mark goal and adversary
    ax.text(4, 4, '★', ha='center', va='center', fontsize=20, color='green')
    ax.text(adv_pos[1], adv_pos[0], '👾', ha='center', va='center', fontsize=16)
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.colorbar(im, ax=ax, label='State Value')

    # Right: Value heatmap
    ax = axes[1]
    ax.set_title('State Value Heatmap', fontsize=12)
    im2 = ax.imshow(value_grid, cmap='RdYlGn')
    for row in range(grid_size):
        for col in range(grid_size):
            ax.text(col, row, f'{value_grid[row,col]:.1f}',
                   ha='center', va='center', fontsize=9)
    plt.colorbar(im2, ax=ax, label='Max Q-value')
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_learning_curve(returns, algorithm='Agent', 
                        save_path='results/learning_curve.png', window=100):
    """Plot smoothed learning curve with raw returns in background"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(returns, alpha=0.2, color='steelblue', label='Raw')
    if len(returns) >= window:
        smoothed = np.convolve(returns, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, color='steelblue', linewidth=2,
                label=f'Moving Avg ({window} eps)')
    ax.set_title(f'{algorithm} Learning Curve', fontsize=14, fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Return')
    ax.legend()
    ax.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Saved: {save_path}")


def dqn_saliency(agent, state, save_path='results/dqn_saliency.png'):
    """
    Saliency map for DQN: measures how much each input feature
    affects the Q-value output using finite differences.
    
    Higher saliency = agent is more sensitive to that input dimension.
    Input dims: [agent_row, agent_col, adv_row, adv_col]
    """
    encoded = agent.encode_state(state)
    base_q  = agent.online_net.forward(encoded.copy())
    epsilon = 0.01
    saliency = np.zeros(len(encoded))

    for i in range(len(encoded)):
        perturbed = encoded.copy()
        perturbed[i] += epsilon
        perturbed_q = agent.online_net.forward(perturbed)
        saliency[i] = np.max(np.abs(perturbed_q - base_q)) / epsilon

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ['Agent Row', 'Agent Col', 'Adv Row', 'Adv Col']
    colors = ['steelblue' if s < max(saliency) else 'coral' for s in saliency]
    bars = ax.bar(labels, saliency, color=colors)
    ax.set_title(f'DQN Saliency Map\nState: {state}', 
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Sensitivity (dQ/dinput)')
    ax.set_xlabel('Input Feature')
    for bar, val in zip(bars, saliency):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.3f}', ha='center', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Saved: {save_path}")


if __name__ == "__main__":
    print("Visualization utilities loaded successfully!")
    print("Available functions:")
    print("  - plot_policy_heatmap(Q_table)")
    print("  - plot_learning_curve(returns, algorithm)")
    print("  - dqn_saliency(agent, state)")