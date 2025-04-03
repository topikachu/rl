import numpy as np
import robot_pb2
from logger_config import get_logger, get_reward_logger
import math

logger = get_logger(__name__)
reward_logger = get_reward_logger()
import torch
class RobocodeEnv:
    DAMAGE_DEALT_SCALE = 15
    DAMAGE_TAKEN_SCALE = 7
    GUN_TURN_IMPROVEMENT_SCALE = 2.0
    GUN_TURN_PENALTY_SCALE = 3.0  # Make the penalty slightly higher than the improvement reward
    ACCURACY_REWARD_SCALE = 5.0
    GUN_TURN_PRECISION_SCALE = 5
    AIM_REWARD_SCALE = 50  # Increased from 5
    STEP_PENALTY = 0.1
    ROBOT_SIZE = 36  # pixels

    def __init__(self, writer):
        self.battlefield_width = 400  # Small battlefield width
        self.battlefield_height = 400  # Small battlefield height
        self.max_distance = np.sqrt(self.battlefield_width**2 + self.battlefield_height**2)
        self.writer = writer
        self.step_count = 0

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
    
        # Calculate damage dealt and taken
        damage_on_enemy = max(0, previous_state.enemyState.energy - current_state.enemyState.energy)
        damage_on_robot = max(0, previous_state.robotState.energy - current_state.robotState.energy)
    
        # Calculate individual reward components
        damage_dealt_reward = damage_on_enemy * self.DAMAGE_DEALT_SCALE
        damage_taken_penalty = -damage_on_robot * self.DAMAGE_TAKEN_SCALE
        gun_turn_reward = self._calculate_gun_turn_reward(previous_state.robotState.gunBearing,
                                                          current_state.robotState.gunBearing
                                                          )
        aim_reward = 0
        fire_penalty = 0
        if previous_action in [2, 3]:  # If the action was firing (small or normal)
            aim_reward, fire_penalty = self._calculate_aim_reward(
                current_state.robotState.gunBearing,
                current_state.enemyState.distance,
                previous_action
            )
    
        step_penalty = -self.STEP_PENALTY
    
        # Sum up all reward components
        total_reward = damage_dealt_reward + damage_taken_penalty + gun_turn_reward + aim_reward + fire_penalty + step_penalty
    
        # Log the reward components
        self.writer.add_scalars('Reward_Components', {
            'damage_dealt': damage_dealt_reward,
            'damage_taken': damage_taken_penalty,
            'gun_turn': gun_turn_reward,
            'aim': aim_reward,
            'fire_penalty': fire_penalty,
            'step_penalty': step_penalty
        }, self.step_count)
        
        self.writer.add_scalar('Total_Reward', total_reward, self.step_count)
        
        self.step_count += 1
    
        # Log the reward components
        reward_logger.debug(f"Step: {self.step_count}, Reward breakdown - "
                            f"damage_dealt: {damage_dealt_reward:.2f}, "
                            f"damage_taken: {damage_taken_penalty:.2f}, "
                            f"gun_turn: {gun_turn_reward:.2f}, "
                            f"aim: {aim_reward:.2f}, "
                            f"fire_penalty: {fire_penalty:.2f}, "
                            f"step_penalty: {step_penalty:.2f}, "
                            f"total: {total_reward:.2f}")
        
        return total_reward
    
    def _calculate_gun_turn_reward(self, last_gun_bearing, current_gun_bearing):
        last_aim_error = abs(last_gun_bearing)
        current_aim_error = abs(current_gun_bearing)
        
        # Calculate the improvement in aiming
        aim_improvement = last_aim_error - current_aim_error
        
        # Base reward for improvement
        if aim_improvement > 0:
            # Reward for turning towards the enemy
            base_reward = aim_improvement * self.GUN_TURN_IMPROVEMENT_SCALE
        else:
            # Penalty for turning away from the enemy
            base_reward = aim_improvement * self.GUN_TURN_PENALTY_SCALE
        
        # Additional reward for accuracy
        accuracy_reward = 0
        if current_aim_error < 10:  # If within 10 degrees of perfect aim
            accuracy_reward = (10 - current_aim_error) * self.ACCURACY_REWARD_SCALE
        
        total_reward = base_reward + accuracy_reward
        
        return total_reward
    
    def _calculate_aim_reward(self, current_gun_bearing, enemy_distance, action):
        # Calculate the angle that would still result in a hit
        angle_tolerance = math.degrees(math.atan2(self.ROBOT_SIZE / 2, enemy_distance))
    
        aim_error_degrees = abs(current_gun_bearing)
    
        aim_reward = 0
        fire_penalty = 0
        
        if aim_error_degrees <= angle_tolerance:
            # The shot would hit the enemy
            aim_reward = (1 - (aim_error_degrees / angle_tolerance)) * self.AIM_REWARD_SCALE
        elif action in [2, 3]:  # Only apply fire penalty if the action is firing (small or normal)
            # The shot would miss
            fire_penalty = -self.AIM_REWARD_SCALE * (aim_error_degrees - angle_tolerance) / (180 - angle_tolerance)
            # Increase penalty for normal fire (action 3) when not aimed well
            if action == 3:
                fire_penalty *= 2
        
        return aim_reward, fire_penalty
    def _normalize_bearing(self, bearing):
        return bearing / 180.0