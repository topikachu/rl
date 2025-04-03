import numpy as np
import robot_pb2
import torch
class RobocodeEnv:
    DAMAGE_DEALT_SCALE = 15
    DAMAGE_TAKEN_SCALE = 7
    GUN_TURN_IMPROVEMENT_SCALE = 0.1
    GUN_TURN_PRECISION_SCALE = 5
    AIM_REWARD_SCALE = 5
    STEP_PENALTY = 0.1

    def __init__(self):
        self.battlefield_width = 400  # Small battlefield width
        self.battlefield_height = 400  # Small battlefield height
        self.max_distance = np.sqrt(self.battlefield_width**2 + self.battlefield_height**2)

    #keep this method for compatibility
    def reset(self):
        pass

    def process_observation(self, game_state):
        return torch.tensor([
            game_state.robotState.gunBearing / 180.0,
            game_state.enemyState.distance / self.max_distance,
            game_state.robotState.energy / 100.0,  # Assuming max energy is 100
            game_state.enemyState.energy / 100.0,
            game_state.robotState.x / self.battlefield_width,
            game_state.robotState.y / self.battlefield_height
        ], dtype=torch.float32)

    def process_observation_batch(self, game_states):
        # Assuming game_states is a list of game state objects
        return torch.stack([self.process_observation(state) for state in game_states])

    def calculate_reward(self, previous_state, previous_action, current_state):
        if previous_state is None or previous_action is None or current_state is None:
            return 0
    
        reward = 0
    
        # Calculate damage dealt and taken
        damage_on_enemy = max(0, previous_state.enemyState.energy - current_state.enemyState.energy)
        damage_on_robot = max(0, previous_state.robotState.energy - current_state.robotState.energy)
    
        # 1. Encourage turning the gun toward the target
        gun_turn_reward = self._calculate_gun_turn_reward(previous_state.robotState.gunBearing,
                                                          current_state.robotState.gunBearing)
        reward += gun_turn_reward
    
        # 2. Reward firing when correctly aimed
        if previous_action in [2, 3]:  # If the action was firing (small or normal)
            aim_reward = self._calculate_aim_reward(current_state.robotState.gunBearing)
            reward += aim_reward
    
        # Existing rewards
        if damage_on_enemy > 0:
            reward += damage_on_enemy * self.DAMAGE_DEALT_SCALE
    
        if damage_on_robot > 0:
            reward -= damage_on_robot * self.DAMAGE_TAKEN_SCALE
        
        reward -= self.STEP_PENALTY  # Small penalty for each step to ask finish quick
        return reward
    
    def _calculate_gun_turn_reward(self, last_gun_bearing, current_gun_bearing):
        bearing_improvement = abs(self._normalize_bearing(last_gun_bearing)) - abs(self._normalize_bearing(current_gun_bearing))
        improvement_reward = bearing_improvement * self.GUN_TURN_IMPROVEMENT_SCALE
    
        precision_reward = 1 - min(abs(self._normalize_bearing(current_gun_bearing)), 1)
        precision_reward *= self.GUN_TURN_PRECISION_SCALE
    
        total_reward = improvement_reward + precision_reward
        return total_reward
    
    def _calculate_aim_reward(self, current_gun_bearing):
        aim_precision = 1 - min(abs(self._normalize_bearing(current_gun_bearing)), 1)
        return aim_precision * self.AIM_REWARD_SCALE

    def _normalize_bearing(self, bearing):
        return bearing / 180.0