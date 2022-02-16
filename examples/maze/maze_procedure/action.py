from dataclasses import dataclass

from maze.maze import Direction

from hrl.action import ProcedureRequest


@dataclass
class GoDirection(ProcedureRequest):
    direction: Direction
