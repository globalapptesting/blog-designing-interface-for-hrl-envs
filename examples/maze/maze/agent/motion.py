from collections import OrderedDict
from typing import TypedDict, Union, Optional

import numpy as np
import numpy.typing as npt
from gym import Space  # type: ignore
from gym.spaces import Dict, Box, Discrete, MultiBinary  # type: ignore

from hrl.action import SwitchAgent
from hrl.agent import Agent, AgentObs
from hrl.exceptions import UnknownAgentAction
from maze.action import MoveForward, MoveBackward
from maze.env_config import MazeEnvConfig
from maze.env_state import MazeEnvState
from maze.exceptions import DirectionNonWalkable
from maze.maze import Direction

MotionAgentState = MazeEnvState
MotionAgentAction = Union[MoveForward, MoveBackward]
MotionAgentRawAction = int


class MotionAgentObs(TypedDict):
    map: npt.NDArray[np.float32]
    position: npt.NDArray[np.float32]
    directions_mask: npt.NDArray[np.float32]


class MotionAgentConfig(TypedDict):
    max_steps: int
    reward_for_right_direction: float
    reward_for_wrong_direction: float


class MotionAgent(
    Agent[
        MazeEnvConfig,
        MazeEnvState,
        MotionAgentConfig,
        MotionAgentState,
        MotionAgentObs,
        MotionAgentRawAction,
        MotionAgentAction,
    ]
):
    NAME = "motion"

    DEFAULTS: MotionAgentConfig = {
        "max_steps": 10,
        "reward_for_right_direction": 1.0,
        "reward_for_wrong_direction": -0.1,
    }

    def __init__(self, config: MotionAgentConfig, env_config: MazeEnvConfig):
        super().__init__(config, env_config)
        self._elapsed_steps: Optional[int] = None

    @staticmethod
    def observation_space(
        config: MotionAgentConfig, env_config: MazeEnvConfig
    ) -> Space:
        map = env_config["map"]
        rows = len(map)
        cols = len(map[0])
        return Dict(
            OrderedDict(
                [
                    ("map", Box(low=0, high=3, shape=(rows, cols))),
                    ("position", Box(low=0, high=max(rows, cols), shape=(2,))),
                    ("directions_mask", MultiBinary(2)),
                ]
            )
        )

    @staticmethod
    def action_space(config: MotionAgentConfig, env_config: MazeEnvConfig) -> Space:
        # MoveForward and MoveBackward
        return Discrete(2)

    def translate_state(self, state: MazeEnvState) -> MotionAgentState:
        return state

    def encode_observation(self, state: MotionAgentState) -> AgentObs:
        available_directions = [
            state.maze.is_direction_walkable(
                state.position, Direction.opposite(state.direction)
            ),
            state.maze.is_direction_walkable(state.position, state.direction),
        ]
        encoded_available_directions = np.array(available_directions)
        return OrderedDict(
            [
                ("map", state.maze.map.astype(dtype=np.float32)),
                ("position", np.array(state.position, dtype=np.float32)),
                ("directions_mask", encoded_available_directions),
            ]
        )  # type: ignore

    def decode_action(
        self, state: MotionAgentState, action: MotionAgentRawAction
    ) -> MotionAgentAction:
        if action == 0:
            direction = Direction.opposite(state.direction)
            if not state.maze.is_direction_walkable(state.position, direction):
                raise DirectionNonWalkable(direction)
            return MoveBackward()
        elif action == 1:
            if not state.maze.is_direction_walkable(state.position, state.direction):
                raise DirectionNonWalkable(state.direction)
            return MoveForward()
        raise UnknownAgentAction(self, action)

    def has_done(self, state: MotionAgentState) -> bool:
        return any(
            [
                self._elapsed_steps >= self.config["max_steps"],
                state.maze.is_intersection(state.position) and self._elapsed_steps > 0,
                not state.maze.is_direction_walkable(state.position, state.direction),
            ]
        )

    def calculate_reward(
        self,
        state: MotionAgentState,
        action: MotionAgentAction,
        new_state: MotionAgentState,
    ) -> float:
        if isinstance(action, MoveForward):
            return self.config["reward_for_right_direction"]
        return self.config["reward_for_wrong_direction"]

    def on_reset(self) -> None:
        self._elapsed_steps = 0

    def on_takes_control(
        self, state: MotionAgentState, action: Optional[SwitchAgent]
    ) -> None:
        self._elapsed_steps = 0

    def on_step(self, action: MotionAgentAction) -> None:
        self._elapsed_steps += 1
