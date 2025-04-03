import numpy as np
import robot_pb2
import torch
class RobocodeEnv:
    def __init__(self):
        self.battlefield_width = 400  # Small battlefield width
        self.battlefield_height = 400  # Small battlefield height


    def process_observation(self, game_state):
        return torch.tensor([
            game_state.enemyState.bearing / 180.0,
            game_state.enemyState.distance / self.max_distance
        ], dtype=torch.float32)

    def process_observation_batch(self, game_states):
        # Assuming game_states is a list of game state objects
        return torch.stack([self.process_observation(state) for state in game_states])

    def calculate_reward(self, previous_state, previous_action, current_state):

        if previous_state is None or previous_action is None or current_state is None:
            return 0
    
        last_enemy = previous_state.enemyState
        last_robot = previous_state.robotState
        enemy_state  = current_state.enemyState
        robot_state = current_state.robotState
    
        reward = 0
    
        # Calculate damage dealt and taken
        damage_dealt = max(0, last_enemy.energy - enemy_state.energy)
        damage_taken = max(0, last_robot.energy - robot_state.energy)
    
        # Reward for hitting the enemy
        if damage_dealt > 0:
            reward += damage_dealt * 15  # Increased reward for successful hits
    
        # Reward for firing
        energy_spent_on_firing = max(0, damage_taken - damage_dealt)
        if energy_spent_on_firing > 0:
            if previous_action == 2:  # Small Fire
                reward += 2  # Small positive reward for small fire
            elif previous_action == 3:  # Normal Fire
                if damage_dealt > 0:
                    reward += damage_dealt * 10  # Larger reward for successful normal fire
                else:
                    reward -= 5  # Penalty for missing with normal fire
    
        # Penalty for getting hit
        if damage_taken > 0:
            reward -= damage_taken * 7  # Increased penalty for taking damage
    
        # Small reward for turning the gun (to encourage exploration)
        if previous_action in [0, 1]:
            reward += 0.1
    
        return reward

