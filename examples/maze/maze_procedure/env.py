from functools import cached_property
from typing import Any, Dict, List, Tuple, Type

from maze.env_config import MazeEnvConfig
from maze.maze import Maze
from maze_procedure.action import GoDirection
from maze_procedure.agent.strategy import StrategyAgent
from maze_procedure.env_state import MazeEnvState
from maze_procedure.procedure.motion import MotionProcedure

from hrl.action import Action
from hrl.agent import Agent, AgentConfig, AgentName, AgentTrigger
from hrl.env import HierarchicalEnv
from hrl.exceptions import UnknownAction
from hrl.procedure import Procedure, ProcedureName


class MazeProcedureEnv(HierarchicalEnv[MazeEnvConfig, MazeEnvState]):
    def __init__(
        self, config: MazeEnvConfig, agent_configs: Dict[AgentName, AgentConfig]
    ):
        super().__init__(config, agent_configs)

        self._maze = Maze()

    @cached_property
    def agents(
        self,
    ) -> Dict[
        AgentName,
        Type[Agent[MazeEnvConfig, MazeEnvState, Any, Any, Any, Any, Any, Any]],
    ]:
        return {StrategyAgent.NAME: StrategyAgent}

    @cached_property
    def procedures(self) -> Dict[ProcedureName, Procedure[MazeEnvState, Any]]:
        return {MotionProcedure.NAME: MotionProcedure()}

    @property
    def initial_agent(self) -> AgentName:
        return StrategyAgent.NAME

    @cached_property
    def procedures_on_action(self) -> List[Tuple[AgentTrigger[Action], ProcedureName]]:
        return [
            (
                lambda name, action: name == StrategyAgent.NAME
                and isinstance(action, GoDirection),
                MotionProcedure.NAME,
            ),
        ]

    def initial_state(self) -> MazeEnvState:
        return MazeEnvState(self._maze, self._maze.start)

    def env_step(self, state: MazeEnvState, action: Action) -> MazeEnvState:
        raise UnknownAction(action)
