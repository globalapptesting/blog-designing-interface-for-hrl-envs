from dataclasses import dataclass

from hrl.action import SwitchAgent, Action
from maze.maze import Direction


@dataclass
class SetDirection(SwitchAgent):
    direction: Direction


@dataclass
class MoveForward(Action):
    pass


@dataclass
class MoveBackward(Action):
    pass
