from typing import Generic, Any

from hrl.action import Action, SwitchAgent
from hrl.agent import AgentRawAction, Agent


class UnknownAction(Exception):
    def __init__(self, action: Action):
        super().__init__(f"Unknown action `{action}`.")


class UnknownAgentAction(Exception, Generic[AgentRawAction]):
    def __init__(
        self,
        agent: Agent[Any, Any, Any, Any, Any, AgentRawAction, Any],
        action: AgentRawAction,
    ):
        super().__init__(f"Unknown action `{action}` for agent `{agent.NAME}`.")


class MissingNextAgent(Exception):
    def __init__(
        self, agent: Agent[Any, Any, Any, Any, Any, Any, Any], action: SwitchAgent
    ):
        super().__init__(
            f"Cannot find any matching trigger for the agent `{agent.NAME}` "
            f"and the action `{action}`."
        )
