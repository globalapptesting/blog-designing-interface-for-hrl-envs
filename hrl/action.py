from abc import ABC
from dataclasses import dataclass

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass(frozen=True)
class Action(ABC):
    pass


@dataclass_json
@dataclass(frozen=True)
class SwitchAgent(Action):
    pass


@dataclass_json
@dataclass(frozen=True)
class NoSwitchAction(SwitchAgent):
    pass


@dataclass_json
@dataclass(frozen=True)
class ProcedureRequest(Action):
    pass
