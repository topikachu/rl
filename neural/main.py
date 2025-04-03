import grpc
from concurrent import futures
import robot_pb2
import robot_pb2_grpc
from robocode_env import RobocodeEnv
from dqn_agent import DQNAgent

class RobotServiceServicer(robot_pb2_grpc.RobotServiceServicer):
    def __init__(self):
        self.env = RobocodeEnv()
        self.agent = DQNAgent(state_size=2, action_size=4)  # Keep this the same for now
        self.previous_state = None
        self.previous_action = None
        self.episode_reward = 0
        self.episode_step = 0
        self.episodes = 0
        self.update_target_every = 5

    def SendState(self, game_state, context):
        # Choose an action based on the current game state

        self.episode_step += 1
    
        # Calculate reward and store experience only if we have a previous state
        if self.previous_state is not None and self.previous_action is not None:
            # Calculate the reward based on the previous action and states
            reward = self.env.calculate_reward(self.previous_state, self.previous_action, game_state)
    
            self.episode_reward += reward
    
            done = False
            self.agent.remember(self.previous_state, self.previous_action, reward, game_state, done)
    
            # Train only if enough samples exist
            if len(self.agent.memory) > 32:
                self.agent.replay(32)


        action = self.agent.act(game_state)

        # Update previous state and action
        self.previous_state = game_state


        self.previous_action = action
    
        # Convert the chosen action to a Robocode action
        robocode_action = self.action_to_robocode(action)
    
        return robot_pb2.Actions(actions=[robocode_action])

    def EndRound(self, request, context):
        if request.result == robot_pb2.RoundResult.WIN:
            print("Round ended: Win")
            final_reward = 2000
        elif request.result == robot_pb2.RoundResult.LOSS:
            print("Round ended: Loss")
            final_reward = -1000
        else:
            print("Round ended: Normal end")
            final_reward = 0

        # Create the final experience tuple
        if self.previous_state is not None and self.previous_action is not None:
            done = True
            self.agent.remember(self.previous_state, self.previous_action, final_reward, self.previous_state, done)

        # Perform a learning step with this final experience
        if len(self.agent.memory) > 32:
            self.agent.replay(32)

        self.episode_reward += final_reward
        print(f"Episode {self.episodes} finished. Total episode steps: {self.episode_step}. Total reward: {self.episode_reward}")

        # Update target model if necessary
        self.episodes += 1
        if self.episodes % self.update_target_every == 0:
            self.agent.update_target_model()
            print(f"Updated target model at episode {self.episodes}")

        # Prepare for the new round
        self.handle_new_round()

        return robot_pb2.google_dot_protobuf_dot_empty__pb2.Empty()

    def handle_new_round(self):
        self.env.reset()
        self.previous_state = None
        self.previous_action = None
        self.episode_reward = 0
        self.episode_step = 0
        print("New round started")

    def action_to_robocode(self, action):
        if action == 0:
            return robot_pb2.Action(action_type=robot_pb2.Action.ActionType.TURN_GUN_LEFT, value=5)
        elif action == 1:
            return robot_pb2.Action(action_type=robot_pb2.Action.ActionType.TURN_GUN_RIGHT, value=5)
        elif action == 2:
            return robot_pb2.Action(action_type=robot_pb2.Action.ActionType.FIRE, value=0.1)  # Small fire
        elif action == 3:
            return robot_pb2.Action(action_type=robot_pb2.Action.ActionType.FIRE, value=1)  # Normal fire

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    robot_pb2_grpc.add_RobotServiceServicer_to_server(RobotServiceServicer(), server)
    server.add_insecure_port('[::]:5000')
    print("Server started on port 5000")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
