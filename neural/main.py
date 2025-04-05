from typing import Optional

import betterproto
from betterproto.lib.std.google.protobuf import Empty as BetterProtoEmpty
from grpclib.server import Server
from grpclib.utils import graceful_exit
from torch.utils.tensorboard import SummaryWriter

import robot
from dqn_agent import DQNAgent
from logger_config import setup_logger, get_logger
from robocode_env import RobocodeEnv, RobocodeGameState

setup_logger()
logger = get_logger(__name__)

EMPTY = BetterProtoEmpty()


class RobotServiceServicer(robot.RobotServiceBase):
    def __init__(self) -> None:
        self.writer: SummaryWriter = SummaryWriter('train-logs')
        self.env: RobocodeEnv = RobocodeEnv(writer=self.writer)
        action_size = len(RobocodeEnv.ActionType)
        self.agent: DQNAgent = DQNAgent(state_size=17, action_size=action_size, env=self.env, writer=self.writer)
        self.previous_state: Optional[RobocodeGameState] = None
        self.previous_action: Optional[int] = None

        self.episode_reward: float = 0
        self.episode_step: int = 0
        self.episodes: int = 0
        self.update_target_every_n_episodes: int = 5
        logger.info("RobotServiceServicer initialized")
        logger.info("TensorBoard writer initialized")

    async def on_event(self, event_wrapper: robot.Event) -> BetterProtoEmpty:

        event_type, actual_event = betterproto.which_one_of(event_wrapper, "eventType")
        if event_type:
            if self.previous_state is not None:
                self.previous_state.events.append(actual_event)
                logger.debug(f"Received event: {event_type}")
            else:
                logger.warning(f"Received event {event_type} but previous_state is None")
        else:
            logger.warning("Received empty event wrapper")
        return EMPTY

    async def start_round(self, _: BetterProtoEmpty) -> BetterProtoEmpty:
        self.handle_new_round()
        return EMPTY

    async def act(self, game_state: robot.GameState) -> robot.Actions:
        current_state = RobocodeGameState(robot_state=game_state.robot_state, enemy=game_state.enemy, events=[])
        action = self.agent.act(current_state)
        robocode_action = self.env.action_to_robocode(action)
        return robot.Actions(actions=[robocode_action])

    async def send_state(self, game_state: robot.GameState) -> robot.Actions:
        self.episode_step += 1
        logger.debug(f"Received game state. Episode step: {self.episode_step}")

        current_state = RobocodeGameState(robot_state=game_state.robot_state, enemy=game_state.enemy, events=[])

        if self.previous_state is not None and self.previous_action is not None:
            reward = self.env.calculate_reward(self.previous_state, self.previous_action, current_state)
            self.episode_reward += reward
            self.agent.remember(self.previous_state, self.previous_action, reward, current_state, done=False)
            logger.debug(f"Calculated reward: {reward}. Total episode reward: {self.episode_reward}")

            if self.episode_step % 4 == 0 and len(self.agent.memory) > 1000:
                self.agent.replay(128)
                logger.debug("Performed replay")

        action = self.agent.act(current_state)
        logger.debug(f"Chosen action: {action}")

        self.previous_state = current_state
        self.previous_action = action

        robocode_action = self.env.action_to_robocode(action)
        logger.debug(f"Converted to Robocode action: {robocode_action}")

        return robot.Actions(actions=[robocode_action])

    async def end_round(self, request: robot.RoundResult) -> BetterProtoEmpty:
        if self.previous_state is None or self.previous_action is None:
            logger.info(
                "Round ended without receiving any state or action. Skipping reward calculation and episode update")
            return EMPTY

        if self.previous_state is not None and self.previous_action is not None:
            reward = self.env.calculate_reward(self.previous_state, self.previous_action, self.previous_state)
            if request.reason == robot.RoundResultReason.WIN:
                reward += 50
                logger.info(f"Round won with reward: {reward}")
            elif request.reason == robot.RoundResultReason.LOSS:
                reward -= 50
                logger.info(f"Round lost with reward: {reward}")
            else:
                logger.info(f"Round ended with unknown reason. Skipping win/loss calculation")
            self.episode_reward += reward

            self.agent.remember(self.previous_state, self.previous_action, reward, self.previous_state, done=True)
            logger.info(f"Calculated reward: {reward}. Total episode reward: {self.episode_reward}")
            self.writer.add_scalar('Episode_Total_Reward', self.episode_reward, self.episodes)

            if self.episode_step % 4 == 0 and len(self.agent.memory) > 1000:
                self.agent.replay(128)
                logger.debug("Performed replay")

        if request.reason == robot.RoundResultReason.WIN:
            self.writer.add_scalar('WinRate', 1, self.episodes)
        else:
            self.writer.add_scalar('WinRate', 0, self.episodes)

        self.episodes += 1
        if self.episodes % self.update_target_every_n_episodes == 0:
            self.agent.update_target_model()
            logger.info(f"Updated target model at episode {self.episodes}")
        self.handle_new_round()
        return EMPTY

    def handle_new_round(self) -> None:
        self.env.reset()
        self.previous_state = None
        self.previous_action = None
        self.episode_reward = 0
        self.episode_step = 0
        logger.info("New round started")


async def serve() -> None:
    servicer = RobotServiceServicer()
    server = Server([servicer])
    with graceful_exit([server]):
        await server.start(port=5001)

        try:
            await server.wait_closed()
        finally:
            servicer.writer.close()

if __name__ == '__main__':
    import asyncio

    asyncio.run(serve())
