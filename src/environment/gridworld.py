"""
Pursuit-Evasion GridWorld Environment
Agent must reach goal while avoiding adversary
"""

import numpy as np

class PursuitEvasionEnv:
    """
    Pursuit-Evasion MDP
    
    State: (agent_row, agent_col, adversary_row, adversary_col)
    Agent goal: Reach goal_state while avoiding adversary
    Adversary: Scripted patrol policy (V1)
    """
    
    def __init__(self, grid_size=5, goal_state=(4,4), adversary_start=(0,4)):
        self.size = grid_size
        self.goal = goal_state
        self.adversary_start = adversary_start
        
        # Actions for agent (adversary uses same)
        self.actions = {
            0: (-1, 0),  # UP
            1: (1, 0),   # DOWN
            2: (0, -1),  # LEFT
            3: (0, 1),   # RIGHT
            4: (0, 0)    # STAY (useful for adversary)
        }
        self.num_actions = 4  # Agent doesn't use STAY
        
        # Rewards
        self.goal_reward = 10.0      # Reach goal
        self.catch_penalty = -10.0   # Caught by adversary
        self.step_cost = -0.1        # Encourage efficiency
        self.proximity_penalty = -0.5 # Being near adversary is risky
        
        # Adversary patrol pattern (clockwise around grid)
        self.adversary_patrol = self._create_patrol_pattern()
    
    def _create_patrol_pattern(self):
        """Create deterministic patrol path for adversary"""
        # Patrol: top-right → bottom-right → bottom-left → top-left → repeat
        patrol = []
        # Right edge going down
        for i in range(self.size):
            patrol.append((i, self.size-1))
        # Bottom edge going left
        for j in range(self.size-2, -1, -1):
            patrol.append((self.size-1, j))
        # Left edge going up
        for i in range(self.size-2, 0, -1):
            patrol.append((i, 0))
        # Top edge going right (back to start)
        for j in range(1, self.size-1):
            patrol.append((0, j))
        return patrol
    
    def get_adversary_next_pos(self, current_adv_pos, timestep=0):
        """Get adversary next position (scripted patrol)"""
        # Find current position in patrol
        if current_adv_pos in self.adversary_patrol:
            idx = self.adversary_patrol.index(current_adv_pos)
            next_idx = (idx + 1) % len(self.adversary_patrol)
            return self.adversary_patrol[next_idx]
        else:
            # If not in patrol, move to nearest patrol point
            return self.adversary_start
    

    def get_all_states(self):
        """Return all valid state tuples (agent_pos, adversary_pos)"""
        states = []
        for ag_i in range(self.size):
            for ag_j in range(self.size):
                for adv_i in range(self.size):
                    for adv_j in range(self.size):
                        states.append(((ag_i, ag_j), (adv_i, adv_j)))
        return states
       
    def is_valid_position(self, pos):
        """Check if position is within grid"""
        row, col = pos
        return 0 <= row < self.size and 0 <= col < self.size
    
    def get_next_state(self, state, action):
        """
        Transition function with adversary movement
        state = (agent_pos, adversary_pos)
        Returns next_state after agent takes action and adversary moves
        """
        agent_pos, adv_pos = state
        
        # Move agent
        ag_row, ag_col = agent_pos
        d_row, d_col = self.actions[action]
        new_agent_pos = (ag_row + d_row, ag_col + d_col)
        
        # Check bounds
        if not self.is_valid_position(new_agent_pos):
            new_agent_pos = agent_pos  # Stay if invalid
        
        # Move adversary (scripted patrol)
        new_adv_pos = self.get_adversary_next_pos(adv_pos)
        
        # Check collision AFTER both moved
        if new_agent_pos == new_adv_pos:
            # Caught! Agent stays where it tried to move
            pass
        
        return (new_agent_pos, new_adv_pos)
    
    def get_reward(self, state, action, next_state):
        """Reward function for pursuit-evasion"""
        agent_pos, adv_pos = state
        next_agent_pos, next_adv_pos = next_state
        
        # Check if caught
        if next_agent_pos == next_adv_pos:
            return self.catch_penalty
        
        # Check if reached goal
        if next_agent_pos == self.goal:
            return self.goal_reward
        
        # Proximity penalty (Manhattan distance)
        dist = abs(next_agent_pos[0] - next_adv_pos[0]) + abs(next_agent_pos[1] - next_adv_pos[1])
        if dist <= 1:  # Adjacent to adversary
            return self.step_cost + self.proximity_penalty
        
        # Normal step cost
        return self.step_cost
    
    def is_terminal(self, state):
        """Terminal if goal reached OR caught"""
        agent_pos, adv_pos = state
        return agent_pos == self.goal or agent_pos == adv_pos