from dataclasses import dataclass

from maze.maze import Direction

from hrl.action import Action, SwitchAgent


@dataclass
class SetDirection(SwitchAgent):
    direction: Direction


@dataclass
class MoveForward(Action):
    pass


@dataclass
class MoveBackward(Action):
    pass
