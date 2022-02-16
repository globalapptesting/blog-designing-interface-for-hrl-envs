from dataclasses import replace

from maze.maze import Direction
from maze_procedure.action import GoDirection
from maze_procedure.env_state import MazeEnvState

from hrl.procedure import Procedure


class MotionProcedure(Procedure[MazeEnvState, GoDirection]):
    NAME = "motion"

    def execute(self, state: MazeEnvState, action: GoDirection) -> MazeEnvState:
        direction = action.direction
        state = self._step(state, direction)
        while not state.maze.is_intersection(
            state.position
        ) and state.maze.is_direction_walkable(state.position, direction):
            state = self._step(state, direction)
        return state

    def _step(self, state: MazeEnvState, direction: Direction) -> MazeEnvState:
        state = replace(
            state, position=state.maze.next_position(state.position, direction)
        )
        return state
