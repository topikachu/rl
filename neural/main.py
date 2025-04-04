from concurrent import futures
import grpc
import robot_pb2
import robot_pb2_grpc
from dqn_agent import DQNAgent
from robocode_env import RobocodeEnv
from torch.utils.tensorboard import SummaryWriter
from logger_config import setup_logger, get_logger

# Set up root logger

setup_logger()
# Get logger for this module
logger = get_logger(__name__)
from torch.utils.tensorboard import SummaryWriter

class RobotServiceServicer(robot_pb2_grpc.RobotServiceServicer):
    def __init__(self):
        self.writer = SummaryWriter('train-logs')
        self.env = RobocodeEnv(writer= self.writer)
        self.agent = DQNAgent(state_size=8, action_size=4, env=self.env, writer=self.writer)
        self.previous_state = None
        self.previous_action = None
        self.episode_reward = 0
        self.episode_step = 0
        self.episodes = 0
        self.update_target_every_n_episodes = 5
        logger.info("RobotServiceServicer initialized")
        logger.info("TensorBoard writer initialized")

    def StartRound(self, _, context):
        self.handle_new_round()
        return robot_pb2.google_dot_protobuf_dot_empty__pb2.Empty()

    def SendState(self, game_state, context):
        self.episode_step += 1
        logger.debug(f"Received game state. Episode step: {self.episode_step}")

        if self.previous_state is not None and self.previous_action is not None:
            reward = self.env.calculate_reward(self.previous_state, self.previous_action, game_state)
            self.episode_reward += reward
            done = False
            self.agent.remember(self.previous_state, self.previous_action, reward, game_state, done)
            logger.debug(f"Calculated reward: {reward}. Total episode reward: {self.episode_reward}")

            if self.episode_step % 4 == 0 and len(self.agent.memory) > 1000:
                self.agent.replay(128)
                logger.debug("Performed replay")

        action = self.agent.act(game_state)
        logger.debug(f"Chosen action: {action}")

        self.previous_state = game_state
        self.previous_action = action

        robocode_action = self.action_to_robocode(action)
        logger.debug(f"Converted to Robocode action: {robocode_action}")

        return robot_pb2.Actions(actions=[robocode_action])

    def EndRound(self, request, context):
        if self.previous_state is None :
            logger.info("Round ended without receiving any state. Skipping reward calculation and episode update")
            return

        if request.result == robot_pb2.RoundResult.WIN:
            logger.info("Round ended: Win")
            final_reward = 2000
        elif request.result == robot_pb2.RoundResult.LOSS:
            logger.info("Round ended: Loss")
            final_reward = -1000
        else:
            logger.info("Round ended: Normal end")
            final_reward = 0

        if self.previous_state is not None and self.previous_action is not None:
            done = True
            self.agent.remember(self.previous_state, self.previous_action, final_reward, self.previous_state, done)
            logger.debug("Added final experience to memory")

        if self.episode_step % 4 == 0 and len(self.agent.memory) > 1000:
            self.agent.replay(128)
            logger.debug("Performed final replay for the episode")

        self.episode_reward += final_reward
        logger.info(f"Episode {self.episodes} finished. Total steps: {self.episode_step}. Total reward: {self.episode_reward}")

        self.writer.add_scalar('Reward/Episode', self.episode_reward, self.episodes)
        self.writer.add_scalar('Steps/Episode', self.episode_step, self.episodes)

        if request.result == robot_pb2.RoundResult.WIN:
            self.writer.add_scalar('WinRate', 1, self.episodes)
        else:
            self.writer.add_scalar('WinRate', 0, self.episodes)

        self.episodes += 1
        if self.episodes % self.update_target_every_n_episodes == 0:
            self.agent.update_target_model()
            logger.info(f"Updated target model at episode {self.episodes}")
        self.handle_new_round()
        return robot_pb2.google_dot_protobuf_dot_empty__pb2.Empty()

    def handle_new_round(self):
        self.env.reset()
        self.previous_state = None
        self.previous_action = None
        self.episode_reward = 0
        self.episode_step = 0
        logger.info("New round started")

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
    servicer = RobotServiceServicer()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    robot_pb2_grpc.add_RobotServiceServicer_to_server(servicer, server)
    server.add_insecure_port('[::]:5001')
    logger.info("Server started on port 5001")
    server.start()
    try:
        server.wait_for_termination()
    finally:
        servicer.writer.close()

if __name__ == '__main__':
    serve()
