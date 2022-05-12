from dataclasses import dataclass

from maze.maze import Direction

from hrl.action import Action, SwitchAgent


@dataclass(frozen=True)
class SetDirection(SwitchAgent):
    direction: Direction


@dataclass(frozen=True)
class MoveForward(Action):
    pass


@dataclass(frozen=True)
class MoveBackward(Action):
    pass
