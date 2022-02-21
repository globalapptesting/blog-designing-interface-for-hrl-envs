from abc import ABC
from dataclasses import dataclass


@dataclass(frozen=True)
class Action(ABC):
    pass


@dataclass(frozen=True)
class SwitchAgent(Action):
    pass


@dataclass(frozen=True)
class NoSwitchAction(SwitchAgent):
    pass


@dataclass(frozen=True)
class ProcedureRequest(Action):
    pass
