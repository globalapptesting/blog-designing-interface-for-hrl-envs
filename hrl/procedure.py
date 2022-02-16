from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from hrl.action import ProcedureRequest
from hrl.env_types import EnvState

ProcedureName = str

ProcedureAction = TypeVar("ProcedureAction", bound=ProcedureRequest)


class Procedure(ABC, Generic[EnvState, ProcedureAction]):
    NAME: ProcedureName

    @abstractmethod
    def execute(self, state: EnvState, action: ProcedureAction) -> EnvState:
        pass
