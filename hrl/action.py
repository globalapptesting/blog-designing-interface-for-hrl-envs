from abc import ABC
from dataclasses import dataclass

from dataclasses_json import DataClassJsonMixin


@dataclass(frozen=True)
class Action(DataClassJsonMixin, ABC):
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
