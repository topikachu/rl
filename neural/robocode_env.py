import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict

import numpy as np
import torch

import robot  # This should be your gRPC generated module
from logger_config import get_logger, get_reward_logger


@dataclass
class RobocodeGameState:
    robot_state: robot.RobotState
    enemy: Optional[robot.ScannedRobotEvent]
    events: List[robot.Event]


logger = get_logger(__name__)
reward_logger = get_reward_logger()

class RobocodeEnv:
    DAMAGE_DEALT_SCALE = 15
    DAMAGE_TAKEN_SCALE = 7
    GUN_TURN_IMPROVEMENT_SCALE = 0.1
    GUN_TURN_PENALTY_SCALE = 0.15  # Make the penalty slightly higher than the improvement reward
    ACCURACY_REWARD_SCALE = 5.0

    STEP_PENALTY = 0.1
    ROBOT_SIZE = 36  # pixels
    MISSING_BEARING = 360  # Use this to represent missing bearing data
    BULLET_MISS_PENALTY = 5  # Adjust this value as needed
    MAX_VELOCITY = 8.0
    MAX_ENERGY = 100.0
    MAX_GUN_HEAT = 3.0
    COLLISION_PENALTY = 5
    FIRING_ACCURACY_REWARD = 10.0
    FIRING_POWER_REWARD_SCALE = 5.0
    FIRING_PENALTY = -2.0
    WALL_HIT_PENALTY = 10


    class ActionType(Enum):
            MOVE_FORWARD_SMALL = 0
            MOVE_FORWARD_MEDIUM = 1
            MOVE_FORWARD_LARGE = 2
            MOVE_BACKWARD_SMALL = 3
            MOVE_BACKWARD_MEDIUM = 4
            MOVE_BACKWARD_LARGE = 5
            TURN_LEFT_SMALL = 6
            TURN_LEFT_MEDIUM = 7
            TURN_LEFT_LARGE = 8
            TURN_RIGHT_SMALL = 9
            TURN_RIGHT_MEDIUM = 10
            TURN_RIGHT_LARGE = 11
            TURN_GUN_LEFT_SMALL = 12
            TURN_GUN_LEFT_MEDIUM = 13
            TURN_GUN_LEFT_LARGE = 14
            TURN_GUN_RIGHT_SMALL = 15
            TURN_GUN_RIGHT_MEDIUM = 16
            TURN_GUN_RIGHT_LARGE = 17
            FIRE_SMALL = 18
            FIRE_MEDIUM = 19
            FIRE_LARGE = 20
            ROTATE_RADAR_LEFT_SMALL = 21
            ROTATE_RADAR_LEFT_MEDIUM = 22
            ROTATE_RADAR_LEFT_LARGE = 23
            ROTATE_RADAR_RIGHT_SMALL = 24
            ROTATE_RADAR_RIGHT_MEDIUM = 25
            ROTATE_RADAR_RIGHT_LARGE = 26
            DO_NOTHING = 27

            def is_fire_action(self) -> bool:
                return self in [self.FIRE_SMALL, self.FIRE_MEDIUM, self.FIRE_LARGE]

    FIRING_REWARD_SCALE = {
        ActionType.FIRE_SMALL: 0.1,
        ActionType.FIRE_MEDIUM: 1,
        ActionType.FIRE_LARGE: 3,
    }
    def __init__(self, writer):
        self.writer = writer
        self.step_count = 0

        self.action_map = {
            self.ActionType.MOVE_FORWARD_SMALL: (robot.ActionActionType.MOVE_FORWARD, 25),
            self.ActionType.MOVE_FORWARD_MEDIUM: (robot.ActionActionType.MOVE_FORWARD, 50),
            self.ActionType.MOVE_FORWARD_LARGE: (robot.ActionActionType.MOVE_FORWARD, 100),
            self.ActionType.MOVE_BACKWARD_SMALL: (robot.ActionActionType.MOVE_BACKWARD, 25),
            self.ActionType.MOVE_BACKWARD_MEDIUM: (robot.ActionActionType.MOVE_BACKWARD, 50),
            self.ActionType.MOVE_BACKWARD_LARGE: (robot.ActionActionType.MOVE_BACKWARD, 100),
            self.ActionType.TURN_LEFT_SMALL: (robot.ActionActionType.TURN_LEFT, 5),
            self.ActionType.TURN_LEFT_MEDIUM: (robot.ActionActionType.TURN_LEFT, 15),
            self.ActionType.TURN_LEFT_LARGE: (robot.ActionActionType.TURN_LEFT, 45),
            self.ActionType.TURN_RIGHT_SMALL: (robot.ActionActionType.TURN_RIGHT, 5),
            self.ActionType.TURN_RIGHT_MEDIUM: (robot.ActionActionType.TURN_RIGHT, 15),
            self.ActionType.TURN_RIGHT_LARGE: (robot.ActionActionType.TURN_RIGHT, 45),
            self.ActionType.TURN_GUN_LEFT_SMALL: (robot.ActionActionType.TURN_GUN_LEFT, 5),
            self.ActionType.TURN_GUN_LEFT_MEDIUM: (robot.ActionActionType.TURN_GUN_LEFT, 15),
            self.ActionType.TURN_GUN_LEFT_LARGE: (robot.ActionActionType.TURN_GUN_LEFT, 45),
            self.ActionType.TURN_GUN_RIGHT_SMALL: (robot.ActionActionType.TURN_GUN_RIGHT, 5),
            self.ActionType.TURN_GUN_RIGHT_MEDIUM: (robot.ActionActionType.TURN_GUN_RIGHT, 15),
            self.ActionType.TURN_GUN_RIGHT_LARGE: (robot.ActionActionType.TURN_GUN_RIGHT, 45),
            self.ActionType.FIRE_SMALL: (robot.ActionActionType.FIRE, 0.1),
            self.ActionType.FIRE_MEDIUM: (robot.ActionActionType.FIRE, 1),
            self.ActionType.FIRE_LARGE: (robot.ActionActionType.FIRE, 3),
            self.ActionType.ROTATE_RADAR_LEFT_SMALL: (robot.ActionActionType.ROTATE_RADAR_LEFT, 5),
            self.ActionType.ROTATE_RADAR_LEFT_MEDIUM: (robot.ActionActionType.ROTATE_RADAR_LEFT, 15),
            self.ActionType.ROTATE_RADAR_LEFT_LARGE: (robot.ActionActionType.ROTATE_RADAR_LEFT, 45),
            self.ActionType.ROTATE_RADAR_RIGHT_SMALL: (robot.ActionActionType.ROTATE_RADAR_RIGHT, 5),
            self.ActionType.ROTATE_RADAR_RIGHT_MEDIUM: (robot.ActionActionType.ROTATE_RADAR_RIGHT, 15),
            self.ActionType.ROTATE_RADAR_RIGHT_LARGE: (robot.ActionActionType.ROTATE_RADAR_RIGHT, 45),
            self.ActionType.DO_NOTHING: (robot.ActionActionType.DO_NOTHING, 0),
        }

    #keep this method for compatibility
    def reset(self):
        pass

    def process_observation(self, game_state: RobocodeGameState) -> torch.Tensor:
        robot_state = game_state.robot_state
        enemy_state = game_state.enemy
        max_distance = np.sqrt(robot_state.battle_field_width ** 2 + robot_state.battle_field_height ** 2)
        # Robot state tensor
        robot_tensor = torch.tensor([
            robot_state.x / robot_state.battle_field_width,
            robot_state.y / robot_state.battle_field_height,
            robot_state.velocity / self.MAX_VELOCITY,
            robot_state.heading / (2 * math.pi),  # Normalize heading (radians)
            robot_state.gun_heading / (2 * math.pi),  # Normalize gun heading (radians)
            robot_state.radar_heading / (2 * math.pi),  # Normalize radar heading (radians)
            robot_state.gun_heat / self.MAX_GUN_HEAT,
            robot_state.gun_turn_remaining / (2 * math.pi),  # Normalize gun turn remaining (radians)
            robot_state.radar_turn_remaining / (2 * math.pi),  # Normalize radar turn remaining (radians)
            robot_state.energy / self.MAX_ENERGY,
        ], dtype=torch.float32)

        # Enemy state tensor
        if enemy_state is None:
            enemy_tensor = torch.tensor([
                -1, -1,  # x, y
                0,  # velocity
                -1,  # heading
                self.MISSING_BEARING,  # bearing
                -1,  # distance
                -1,  # energy
            ], dtype=torch.float32)
        else:
            enemy_tensor = torch.tensor([
                enemy_state.x / robot_state.battle_field_width,
                enemy_state.y / robot_state.battle_field_height,
                enemy_state.velocity / self.MAX_VELOCITY,
                enemy_state.heading / (2 * math.pi),  # Normalize heading (radians)
                self._normalize_bearing(enemy_state.bearing),
                enemy_state.distance / max_distance,
                enemy_state.energy / self.MAX_ENERGY,
            ], dtype=torch.float32)

        # Combine robot and enemy tensors
        return torch.cat([robot_tensor, enemy_tensor])

    def _normalize_bearing(self, bearing):
        if bearing == self.MISSING_BEARING:
            return 2  # Use a value outside the normal range (-1 to 1)
        return bearing / 180.0  # This will keep the range between -1 and 1

    def process_observation_batch(self, game_states: List[RobocodeGameState]) -> torch.Tensor:
        # Assuming game_states is a list of game state objects
        return torch.stack([self.process_observation(state) for state in game_states])

    def calculate_reward(self, previous_state: RobocodeGameState, previous_action: int,
                         current_state: RobocodeGameState) -> float:
        if previous_state is None or previous_action is None or current_state is None:
            return 0

        reward_breakdown = {
            'penaltyByHitByBullet': 0,
            'awardByBulletHit': 0,
            'penaltyByHitWall': 0,
            'awardByWinning': 0,
            'penaltyByDying': 0,
            'gunTurnReward': 0,
            'penaltyByBulletMissed': 0,
            'penaltyByCollision': 0,
            'stepPenalty': -self.STEP_PENALTY,
            'firingAccuracyReward': 0,
            'firingPowerReward': 0,
            'firingPenalty': 0
        }

        # Extract robot states
        prev_robot_state = previous_state.robot_state
        curr_robot_state = current_state.robot_state

        # Handle enemy information
        prev_enemy_state = previous_state.enemy
        curr_enemy_state = current_state.enemy

        # Calculate gun bearings
        previous_gun_bearing = self._calculate_gun_bearing(prev_robot_state, prev_enemy_state)
        current_gun_bearing = self._calculate_gun_bearing(curr_robot_state, curr_enemy_state)

        # Calculate gun turn reward
        if previous_gun_bearing is not None and current_gun_bearing is not None and curr_enemy_state is not None:
            reward_breakdown['gunTurnReward'] = self._calculate_gun_turn_reward(
                previous_gun_bearing,
                current_gun_bearing,
                curr_enemy_state.distance
            )

            firing_rewards = self._calculate_firing_reward(
                current_gun_bearing,
                curr_enemy_state.distance,
                previous_action
            )
            for award, value in firing_rewards.items():
                reward_breakdown[award] += value

        # Process events
        for event in previous_state.events:
            if isinstance(event, robot.HitByBulletEvent):
                bullet_power = event.bullet.power
                reward_breakdown['penaltyByHitByBullet'] -= self.DAMAGE_TAKEN_SCALE * bullet_power
            elif isinstance(event, robot.BulletHitEvent):
                bullet_power = event.bullet.power
                reward_breakdown['awardByBulletHit'] += self.DAMAGE_DEALT_SCALE * bullet_power
            elif isinstance(event, robot.HitWallEvent):
                reward_breakdown['penaltyByHitWall'] -= self.WALL_HIT_PENALTY
            elif isinstance(event, robot.BulletMissedEvent):
                bullet_power = event.bullet.power
                reward_breakdown['penaltyByBulletMissed'] -= self.BULLET_MISS_PENALTY * bullet_power
            elif isinstance(event, robot.HitRobotEvent):
                # Add penalty for collision
                collision_penalty = self.COLLISION_PENALTY
                if event.at_fault:
                    # If the robot is at fault, increase the penalty
                    collision_penalty *= 2
                reward_breakdown['penaltyByCollision'] -= collision_penalty
            else:
                logger.warning(f"Unhandled event: {event}")

        total_reward = sum(reward_breakdown.values())

        # Log reward components
        self.log_reward_components(reward_breakdown, total_reward)

        return total_reward

    def _calculate_gun_bearing(self, robot_state: robot.RobotState, enemy_state: Optional[robot.ScannedRobotEvent]) -> \
    Optional[float]:
        if enemy_state and enemy_state.bearing != self.MISSING_BEARING:
            return self.calculate_bearing_from_gun(
                robot_state.heading,
                enemy_state.bearing,
                robot_state.gun_heading
            )
        return None

    def log_reward_components(self, reward_breakdown, total_reward):
        self.writer.add_scalars('Reward_Components', reward_breakdown, self.step_count)
        self.writer.add_scalar('Total_Reward', total_reward, self.step_count)

        self.step_count += 1

        reward_logger.debug(f"Step: {self.step_count}, Reward breakdown - " +
                            ", ".join([f"{k}: {v:.2f}" for k, v in reward_breakdown.items()]) +
                            f", total: {total_reward:.2f}")

    def _calculate_gun_turn_reward(self, last_gun_bearing, current_gun_bearing, enemy_distance):
        last_aim_error = abs(last_gun_bearing)
        current_aim_error = abs(current_gun_bearing)

        # Calculate the angle tolerance based on robot size and enemy distance
        angle_tolerance = math.degrees(math.atan2(self.ROBOT_SIZE / 2, enemy_distance))

        # Calculate the improvement in aiming
        aim_improvement = last_aim_error - current_aim_error

        # Base reward for improvement
        if aim_improvement > 0:
            # Reward for turning towards the enemy
            base_reward = aim_improvement * self.GUN_TURN_IMPROVEMENT_SCALE
        elif aim_improvement < 0:
            # Penalty for turning away from the enemy
            base_reward = aim_improvement * self.GUN_TURN_PENALTY_SCALE
        else:
            # No movement
            if current_aim_error <= angle_tolerance:
                # Reward for maintaining good aim
                base_reward = self.ACCURACY_REWARD_SCALE
            else:
                # Penalty for not improving when aim is poor
                base_reward = -self.GUN_TURN_PENALTY_SCALE * 5

        # Additional reward for accuracy
        if current_aim_error <= angle_tolerance:
            accuracy_reward = (angle_tolerance - current_aim_error) * self.ACCURACY_REWARD_SCALE
        else:
            accuracy_reward = 0

        total_reward = base_reward + accuracy_reward

        return total_reward

    def _calculate_firing_reward(self, current_gun_bearing, enemy_distance, action) -> Dict[str, float]:
        firing_reward = {
            'firingAccuracyReward': 0,
            'firingPowerReward': 0,
            'firingPenalty': 0
        }

        try:
            action_type: RobocodeEnv.ActionType = self.ActionType(action)
        except ValueError:
            logger.warning(f"Invalid action {action} passed to _calculate_firing_reward")
            return firing_reward  # Return default rewards

        if action_type.is_fire_action():
            current_aim_error = abs(current_gun_bearing)
            angle_tolerance = math.degrees(math.atan2(self.ROBOT_SIZE / 2, enemy_distance))
            if current_aim_error <= angle_tolerance:
                # Reward for firing accurately
                firing_reward['firingAccuracyReward'] = self.FIRING_ACCURACY_REWARD
                # Additional reward based on firing power
                firing_reward['firingPowerReward'] = self.FIRING_POWER_REWARD_SCALE * RobocodeEnv.FIRING_REWARD_SCALE[action_type]
            else:
                # Penalty for firing inaccurately
                firing_reward['firingPenalty'] = self.FIRING_PENALTY

        return firing_reward

    def calculate_bearing_from_gun(self, robot_heading, enemy_bearing, gun_heading):
        absolute_bearing = (robot_heading + enemy_bearing) % 360
        bearing_from_gun = self._normalize_angle(absolute_bearing - gun_heading)
        return bearing_from_gun

    def _normalize_angle(self, angle):
        return ((angle + 180) % 360) - 180

    def action_to_robocode(self, action: int) -> robot.Action:
        try:
            action_enum = self.ActionType(action)
        except ValueError:
            raise ValueError(
                f"Invalid action: {action}. Action must be an integer between 0 and {len(self.ActionType) - 1}.")

        if action_enum not in self.action_map:
            raise ValueError(f"Invalid action: {action_enum}. Action not found in action map.")

        action_type, value = self.action_map[action_enum]
        return robot.Action(action_type=action_type, value=value)
