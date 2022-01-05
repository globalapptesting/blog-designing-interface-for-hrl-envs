from dataclasses import replace
from functools import cached_property
from typing import Dict, Tuple, Type, Optional, List, Any

from hrl.action import Action
from hrl.agent import Agent, AgentTrigger, AgentName, AgentConfig
from hrl.env import HierarchicalEnv
from hrl.exceptions import UnknownAction
from maze.action import MoveForward, MoveBackward, SetDirection
from maze.agent.motion import MotionAgent
from maze.agent.strategy import StrategyAgent
from maze.env_config import MazeEnvConfig
from maze.env_state import MazeEnvState
from maze.maze import Maze, Direction


class MazeEnv(HierarchicalEnv[MazeEnvConfig, MazeEnvState]):
    def __init__(
        self, config: MazeEnvConfig, agent_configs: Dict[AgentName, AgentConfig]
    ):
        super().__init__(config, agent_configs)

        self._maze = Maze()

    @cached_property
    def agents(
        self,
    ) -> Dict[
        AgentName, Type[Agent[MazeEnvConfig, MazeEnvState, Any, Any, Any, Any, Any]]
    ]:
        return {StrategyAgent.NAME: StrategyAgent, MotionAgent.NAME: MotionAgent}

    @property
    def initial_agent(self) -> AgentName:
        return StrategyAgent.NAME

    @cached_property
    def transitions_on_done(self) -> Dict[AgentName, Optional[AgentName]]:
        return {
            StrategyAgent.NAME: None,
            MotionAgent.NAME: StrategyAgent.NAME,
        }

    @cached_property
    def transitions_on_action(self) -> List[Tuple[AgentTrigger[Action], AgentName]]:
        return [
            (
                lambda name, action: name == StrategyAgent.NAME
                and isinstance(action, SetDirection),
                MotionAgent.NAME,
            ),
        ]

    def initial_state(self) -> MazeEnvState:
        return MazeEnvState(self._maze, self._maze.start, Direction.LEFT)

    def env_step(self, state: MazeEnvState, action: Action) -> MazeEnvState:
        # TODO TWr This could be nicely refactored with structural pattern matching.
        if isinstance(action, SetDirection):
            state = replace(state, direction=action.direction)
        elif isinstance(action, MoveForward):
            new_position = state.maze.next_position(state.position, state.direction)
            state = replace(state, position=new_position)
        elif isinstance(action, MoveBackward):
            if state.maze.is_intersection(state.position):
                # Don't move backward on an intersection.
                new_position = state.position
            else:
                new_position = self._prev_state.position  # type: ignore
            state = replace(state, position=new_position)
        else:
            raise UnknownAction(action)
        return state
