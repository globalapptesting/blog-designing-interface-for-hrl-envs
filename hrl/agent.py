from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, Protocol, TypeVar

from gym import Space  # type: ignore

from hrl.action import Action, SwitchAgent
from hrl.env_types import EnvConfig, EnvState

AgentName = str
AgentConfig = TypeVar("AgentConfig")

AgentAction_contra = TypeVar("AgentAction_contra", bound=Action, contravariant=True)


class AgentTrigger(Protocol[AgentAction_contra]):
    @abstractmethod
    def __call__(self, name: AgentName, action: AgentAction_contra) -> bool:
        pass


AgentState = TypeVar("AgentState")
AgentObs = TypeVar("AgentObs")
AgentRawAction = TypeVar("AgentRawAction")
AgentAction = TypeVar("AgentAction", bound=Action)
SwitchAgentAction = TypeVar("SwitchAgentAction", bound=SwitchAgent)


class Agent(
    ABC,
    Generic[
        EnvConfig,
        EnvState,
        AgentConfig,
        AgentState,
        AgentObs,
        AgentRawAction,
        AgentAction,
        SwitchAgentAction,
    ],
):
    NAME: AgentName

    def __init__(self, config: AgentConfig, env_config: EnvConfig):
        self.config = config
        self.env_config = env_config

    @staticmethod
    @abstractmethod
    def observation_space(config: AgentConfig, env_config: EnvConfig) -> Space:
        pass

    @staticmethod
    @abstractmethod
    def action_space(config: AgentConfig, env_config: EnvConfig) -> Space:
        pass

    @abstractmethod
    def translate_state(self, state: EnvState) -> AgentState:
        """
        Translates the environment state into the agent state. You can use it to narrow
        down what is visible to the agent.
        """
        pass

    @abstractmethod
    def encode_observation(self, state: AgentState) -> AgentObs:
        """
        Transforms the agent state into the observation by encoding it.
        """
        pass

    @abstractmethod
    def decode_action(self, state: AgentState, action: AgentRawAction) -> AgentAction:
        """
        Transforms a raw action (i.e. model's output) into a well-defined action,
        either directly related to the environment or to some other abstraction
        (e.g. agent switching).
        """
        pass

    @abstractmethod
    def has_done(self, state: AgentState) -> bool:
        """
        Whether the agent has done. It can use its current `state` or context from
        `self`.
        """
        pass

    @abstractmethod
    def calculate_reward(
        self, state: AgentState, action: AgentAction, new_state: AgentState
    ) -> float:
        """
        Calculates a reward for the agent (a reward for being in a `state`, taking an
        `action` and landing in a `new_state`).
        """
        pass

    def info(
        self, state: AgentState, action: AgentAction, new_state: AgentState
    ) -> Dict[str, Any]:
        """
        Anything the agent wants to store in `info` structure. This will be stored under
        the agent ID's key.
        """
        return {}

    def on_reset(self) -> None:
        """
        Called once on each new environment episode. It is always called before any call
        to `encode_observation` or `decode_action`.
        """
        pass

    def on_takes_control(
        self, state: EnvState, action: Optional[SwitchAgentAction]
    ) -> EnvState:
        """
        Called when the agent takes control. The `state` represents the current
        environment state and the `action` represents the cause of the agent change,
        in case another agent initiated the change. This might be used to set
        a goal for the agent.
        """
        return state

    def on_step(self, action: AgentAction) -> None:
        """
        Called on each step of the agent (before the action is decoded and the
        observation is encoded).
        """
        pass

    def on_gives_control(self, action: Optional[SwitchAgent]) -> None:
        """
        Called when the agent gives back control (e.g. when it's done or generated
        an agent switch action).
        """
        pass
