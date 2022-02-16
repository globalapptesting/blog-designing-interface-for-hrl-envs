from typing import Any, Generic

from hrl.action import Action, ProcedureRequest, SwitchAgent
from hrl.agent import Agent, AgentRawAction


class UnknownAction(Exception):
    def __init__(self, action: Action):
        super().__init__(f"Unknown action `{action}`.")


class UnknownAgentAction(Exception, Generic[AgentRawAction]):
    def __init__(
        self,
        agent: Agent[Any, Any, Any, Any, Any, AgentRawAction, Any, Any],
        action: AgentRawAction,
    ):
        super().__init__(f"Unknown action `{action}` for agent `{agent.NAME}`.")


class MissingNextAgent(Exception):
    def __init__(
        self, agent: Agent[Any, Any, Any, Any, Any, Any, Any, Any], action: SwitchAgent
    ):
        super().__init__(
            f"Cannot find any matching trigger for the agent `{agent.NAME}` "
            f"and the action `{action}`."
        )


class MissingProcedure(Exception):
    def __init__(
        self,
        agent: Agent[Any, Any, Any, Any, Any, Any, Any, Any],
        action: ProcedureRequest,
    ):
        super().__init__(
            f"Cannot find any matching procedure for the agent `{agent.NAME}` "
            f"and the action `{action}`."
        )
