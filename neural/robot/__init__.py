# Generated by the protocol buffer compiler.  DO NOT EDIT!
# sources: robot.proto
# plugin: python-betterproto
# This file has been @generated

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Optional,
)

import betterproto
import betterproto.lib.google.protobuf as betterproto_lib_google_protobuf
import grpclib
from betterproto.grpc.grpclib_server import ServiceBase


if TYPE_CHECKING:
    import grpclib.server
    from betterproto.grpc.grpclib_client import MetadataLike
    from grpclib.metadata import Deadline


class RoundResultReason(betterproto.Enum):
    UNKNOWN = 0
    WIN = 1
    LOSS = 2


class ActionActionType(betterproto.Enum):
    MOVE_FORWARD = 0
    MOVE_BACKWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    TURN_GUN_LEFT = 4
    TURN_GUN_RIGHT = 5
    FIRE = 6
    ROTATE_RADAR_LEFT = 7
    ROTATE_RADAR_RIGHT = 8
    DO_NOTHING = 9


@dataclass(eq=False, repr=False)
class GameState(betterproto.Message):
    robot_state: "RobotState" = betterproto.message_field(1)
    enemy: "ScannedRobotEvent" = betterproto.message_field(2)


@dataclass(eq=False, repr=False)
class RoundResult(betterproto.Message):
    reason: "RoundResultReason" = betterproto.enum_field(1)


@dataclass(eq=False, repr=False)
class Event(betterproto.Message):
    bullet_hit: "BulletHitEvent" = betterproto.message_field(1, group="eventType")
    bullet_hit_bullet: "BulletHitBulletEvent" = betterproto.message_field(
        2, group="eventType"
    )
    bullet_missed: "BulletMissedEvent" = betterproto.message_field(3, group="eventType")
    hit_by_bullet: "HitByBulletEvent" = betterproto.message_field(4, group="eventType")
    hit_robot: "HitRobotEvent" = betterproto.message_field(5, group="eventType")
    hit_wall: "HitWallEvent" = betterproto.message_field(6, group="eventType")
    robot_death: "RobotDeathEvent" = betterproto.message_field(7, group="eventType")


@dataclass(eq=False, repr=False)
class BulletHitEvent(betterproto.Message):
    name: str = betterproto.string_field(1)
    energy: float = betterproto.double_field(2)
    bullet: "Bullet" = betterproto.message_field(3)


@dataclass(eq=False, repr=False)
class BulletHitBulletEvent(betterproto.Message):
    bullet: "Bullet" = betterproto.message_field(1)
    hit_bullet: "Bullet" = betterproto.message_field(2)


@dataclass(eq=False, repr=False)
class BulletMissedEvent(betterproto.Message):
    bullet: "Bullet" = betterproto.message_field(1)


@dataclass(eq=False, repr=False)
class HitByBulletEvent(betterproto.Message):
    bearing: float = betterproto.double_field(1)
    bullet: "Bullet" = betterproto.message_field(2)


@dataclass(eq=False, repr=False)
class HitRobotEvent(betterproto.Message):
    robot_name: str = betterproto.string_field(1)
    bearing: float = betterproto.double_field(2)
    energy: float = betterproto.double_field(3)
    at_fault: bool = betterproto.bool_field(4)


@dataclass(eq=False, repr=False)
class HitWallEvent(betterproto.Message):
    bearing: float = betterproto.double_field(1)


@dataclass(eq=False, repr=False)
class RobotDeathEvent(betterproto.Message):
    robot_name: str = betterproto.string_field(1)


@dataclass(eq=False, repr=False)
class Bullet(betterproto.Message):
    heading_radians: float = betterproto.double_field(1)
    x: float = betterproto.double_field(2)
    y: float = betterproto.double_field(3)
    power: float = betterproto.double_field(4)
    owner_name: str = betterproto.string_field(5)
    victim_name: str = betterproto.string_field(6)
    is_active: bool = betterproto.bool_field(7)
    bullet_id: int = betterproto.int32_field(8)


@dataclass(eq=False, repr=False)
class RobotState(betterproto.Message):
    x: float = betterproto.double_field(1)
    """Robot Position & Movement"""

    y: float = betterproto.double_field(2)
    velocity: float = betterproto.double_field(3)
    heading: float = betterproto.double_field(4)
    gun_heading: float = betterproto.double_field(5)
    """Gun & Radar"""

    radar_heading: float = betterproto.double_field(6)
    gun_heat: float = betterproto.double_field(7)
    gun_turn_remaining: float = betterproto.double_field(8)
    radar_turn_remaining: float = betterproto.double_field(9)
    energy: float = betterproto.double_field(10)
    """Game State"""

    battle_field_width: float = betterproto.double_field(11)
    battle_field_height: float = betterproto.double_field(12)
    round_num: int = betterproto.int32_field(13)
    time: int = betterproto.int64_field(14)


@dataclass(eq=False, repr=False)
class ScannedRobotEvent(betterproto.Message):
    x: float = betterproto.double_field(1)
    """Enemy Position & Movement"""

    y: float = betterproto.double_field(2)
    velocity: float = betterproto.double_field(3)
    heading: float = betterproto.double_field(4)
    bearing: float = betterproto.double_field(5)
    distance: float = betterproto.double_field(6)
    energy: float = betterproto.double_field(7)
    """Enemy Stats"""

    time: int = betterproto.int64_field(8)


@dataclass(eq=False, repr=False)
class Action(betterproto.Message):
    action_type: "ActionActionType" = betterproto.enum_field(1)
    value: float = betterproto.double_field(2)


@dataclass(eq=False, repr=False)
class Actions(betterproto.Message):
    actions: List["Action"] = betterproto.message_field(1)


class RobotServiceStub(betterproto.ServiceStub):
    async def send_state(
        self,
        game_state: "GameState",
        *,
        timeout: Optional[float] = None,
        deadline: Optional["Deadline"] = None,
        metadata: Optional["MetadataLike"] = None
    ) -> "Actions":
        return await self._unary_unary(
            "/robot.RobotService/SendState",
            game_state,
            Actions,
            timeout=timeout,
            deadline=deadline,
            metadata=metadata,
        )

    async def act(
        self,
        game_state: "GameState",
        *,
        timeout: Optional[float] = None,
        deadline: Optional["Deadline"] = None,
        metadata: Optional["MetadataLike"] = None
    ) -> "Actions":
        return await self._unary_unary(
            "/robot.RobotService/Act",
            game_state,
            Actions,
            timeout=timeout,
            deadline=deadline,
            metadata=metadata,
        )

    async def on_event(
        self,
        event: "Event",
        *,
        timeout: Optional[float] = None,
        deadline: Optional["Deadline"] = None,
        metadata: Optional["MetadataLike"] = None
    ) -> "betterproto_lib_google_protobuf.Empty":
        return await self._unary_unary(
            "/robot.RobotService/OnEvent",
            event,
            betterproto_lib_google_protobuf.Empty,
            timeout=timeout,
            deadline=deadline,
            metadata=metadata,
        )

    async def end_round(
        self,
        round_result: "RoundResult",
        *,
        timeout: Optional[float] = None,
        deadline: Optional["Deadline"] = None,
        metadata: Optional["MetadataLike"] = None
    ) -> "betterproto_lib_google_protobuf.Empty":
        return await self._unary_unary(
            "/robot.RobotService/EndRound",
            round_result,
            betterproto_lib_google_protobuf.Empty,
            timeout=timeout,
            deadline=deadline,
            metadata=metadata,
        )

    async def start_round(
        self,
        betterproto_lib_google_protobuf_empty: "betterproto_lib_google_protobuf.Empty",
        *,
        timeout: Optional[float] = None,
        deadline: Optional["Deadline"] = None,
        metadata: Optional["MetadataLike"] = None
    ) -> "betterproto_lib_google_protobuf.Empty":
        return await self._unary_unary(
            "/robot.RobotService/StartRound",
            betterproto_lib_google_protobuf_empty,
            betterproto_lib_google_protobuf.Empty,
            timeout=timeout,
            deadline=deadline,
            metadata=metadata,
        )


class RobotServiceBase(ServiceBase):

    async def send_state(self, game_state: "GameState") -> "Actions":
        raise grpclib.GRPCError(grpclib.const.Status.UNIMPLEMENTED)

    async def act(self, game_state: "GameState") -> "Actions":
        raise grpclib.GRPCError(grpclib.const.Status.UNIMPLEMENTED)

    async def on_event(self, event: "Event") -> "betterproto_lib_google_protobuf.Empty":
        raise grpclib.GRPCError(grpclib.const.Status.UNIMPLEMENTED)

    async def end_round(
        self, round_result: "RoundResult"
    ) -> "betterproto_lib_google_protobuf.Empty":
        raise grpclib.GRPCError(grpclib.const.Status.UNIMPLEMENTED)

    async def start_round(
        self,
        betterproto_lib_google_protobuf_empty: "betterproto_lib_google_protobuf.Empty",
    ) -> "betterproto_lib_google_protobuf.Empty":
        raise grpclib.GRPCError(grpclib.const.Status.UNIMPLEMENTED)

    async def __rpc_send_state(
        self, stream: "grpclib.server.Stream[GameState, Actions]"
    ) -> None:
        request = await stream.recv_message()
        response = await self.send_state(request)
        await stream.send_message(response)

    async def __rpc_act(
        self, stream: "grpclib.server.Stream[GameState, Actions]"
    ) -> None:
        request = await stream.recv_message()
        response = await self.act(request)
        await stream.send_message(response)

    async def __rpc_on_event(
        self,
        stream: "grpclib.server.Stream[Event, betterproto_lib_google_protobuf.Empty]",
    ) -> None:
        request = await stream.recv_message()
        response = await self.on_event(request)
        await stream.send_message(response)

    async def __rpc_end_round(
        self,
        stream: "grpclib.server.Stream[RoundResult, betterproto_lib_google_protobuf.Empty]",
    ) -> None:
        request = await stream.recv_message()
        response = await self.end_round(request)
        await stream.send_message(response)

    async def __rpc_start_round(
        self,
        stream: "grpclib.server.Stream[betterproto_lib_google_protobuf.Empty, betterproto_lib_google_protobuf.Empty]",
    ) -> None:
        request = await stream.recv_message()
        response = await self.start_round(request)
        await stream.send_message(response)

    def __mapping__(self) -> Dict[str, grpclib.const.Handler]:
        return {
            "/robot.RobotService/SendState": grpclib.const.Handler(
                self.__rpc_send_state,
                grpclib.const.Cardinality.UNARY_UNARY,
                GameState,
                Actions,
            ),
            "/robot.RobotService/Act": grpclib.const.Handler(
                self.__rpc_act,
                grpclib.const.Cardinality.UNARY_UNARY,
                GameState,
                Actions,
            ),
            "/robot.RobotService/OnEvent": grpclib.const.Handler(
                self.__rpc_on_event,
                grpclib.const.Cardinality.UNARY_UNARY,
                Event,
                betterproto_lib_google_protobuf.Empty,
            ),
            "/robot.RobotService/EndRound": grpclib.const.Handler(
                self.__rpc_end_round,
                grpclib.const.Cardinality.UNARY_UNARY,
                RoundResult,
                betterproto_lib_google_protobuf.Empty,
            ),
            "/robot.RobotService/StartRound": grpclib.const.Handler(
                self.__rpc_start_round,
                grpclib.const.Cardinality.UNARY_UNARY,
                betterproto_lib_google_protobuf.Empty,
                betterproto_lib_google_protobuf.Empty,
            ),
        }
