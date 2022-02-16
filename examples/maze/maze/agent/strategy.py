from collections import OrderedDict
from typing import Optional, TypedDict

import numpy as np
import numpy.typing as npt
from gym import Space  # type: ignore
from gym.spaces import Box, Dict, Discrete, MultiBinary  # type: ignore
from maze.action import SetDirection
from maze.env_config import MazeEnvConfig
from maze.env_state import MazeEnvState
from maze.exceptions import DirectionNonWalkable
from maze.maze import Direction

from hrl.action import NoSwitchAction
from hrl.agent import Agent, AgentConfig
from hrl.exceptions import UnknownAgentAction

StrategyAgentState = MazeEnvState
StrategyAgentAction = SetDirection
StrategyAgentRawAction = int
StrategySwitchAgentAction = NoSwitchAction


class StrategyAgentObs(TypedDict):
    map: npt.NDArray[np.float32]
    position: npt.NDArray[np.float32]
    directions_mask: npt.NDArray[np.float32]


class StrategyAgentConfig(TypedDict):
    max_steps: int
    reward_for_reaching_goal: float


class StrategyAgent(
    Agent[
        MazeEnvConfig,
        MazeEnvState,
        StrategyAgentConfig,
        StrategyAgentState,
        StrategyAgentObs,
        StrategyAgentRawAction,
        StrategyAgentAction,
        StrategySwitchAgentAction,
    ]
):
    NAME = "strategy"

    DEFAULTS: StrategyAgentConfig = {"max_steps": 20, "reward_for_reaching_goal": 1.0}

    def __init__(self, config: StrategyAgentConfig, env_config: MazeEnvConfig):
        super().__init__(config, env_config)
        self._elapsed_steps: Optional[int] = None

    @staticmethod
    def observation_space(config: AgentConfig, env_config: MazeEnvConfig) -> Space:
        map = env_config["map"]
        rows = len(map)
        cols = len(map[0])
        return Dict(
            OrderedDict(
                [
                    ("map", Box(low=0, high=3, shape=(rows, cols))),
                    ("position", Box(low=0, high=max(rows, cols), shape=(2,))),
                    ("directions_mask", MultiBinary(len(Direction))),
                ]
            )
        )

    @staticmethod
    def action_space(config: AgentConfig, env_config: MazeEnvConfig) -> Space:
        return Discrete(len(Direction))

    def translate_state(self, state: MazeEnvState) -> StrategyAgentState:
        return state

    def encode_observation(self, state: StrategyAgentState) -> StrategyAgentObs:
        available_directions = state.maze.walkable_directions(state.position)
        encoded_available_directions = np.zeros(
            shape=(
                len(
                    Direction,
                )
            ),
            dtype=np.float32,
        )
        for direction in available_directions:
            encoded_available_directions[direction.value] = 1.0
        return OrderedDict(
            [
                ("map", state.maze.map.astype(dtype=np.float32)),
                ("position", np.array(state.position, dtype=np.float32)),
                ("directions_mask", encoded_available_directions),
            ]
        )  # type: ignore

    def decode_action(
        self, state: StrategyAgentState, action: StrategyAgentRawAction
    ) -> StrategyAgentAction:
        try:
            direction = Direction(action)
        except ValueError:
            raise UnknownAgentAction(self, action)
        if not state.maze.is_direction_walkable(state.position, direction):
            raise DirectionNonWalkable(direction)
        return SetDirection(direction)

    def has_done(self, state: StrategyAgentState) -> bool:
        return any(
            [
                self._elapsed_steps >= self.config["max_steps"],
                state.maze.goal == state.position,
            ]
        )

    def calculate_reward(
        self,
        state: StrategyAgentState,
        action: StrategyAgentAction,
        new_state: StrategyAgentState,
    ) -> float:
        goal_position = state.maze.goal
        if new_state.position == goal_position:
            return self.config["reward_for_reaching_goal"]
        return 0.0

    def on_reset(self) -> None:
        self._elapsed_steps = 0

    def on_step(self, action: StrategyAgentAction) -> None:
        self._elapsed_steps += 1
