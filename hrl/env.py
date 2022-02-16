import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import cached_property
from typing import Any, Dict, Generic, List, Optional, Tuple, Type

from gym import Space  # type: ignore
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

from hrl.action import Action, ProcedureRequest, SwitchAgent
from hrl.agent import Agent, AgentName, AgentTrigger
from hrl.env_types import EnvConfig, EnvState
from hrl.exceptions import MissingNextAgent, MissingProcedure
from hrl.procedure import Procedure, ProcedureName

LOG = logging.getLogger(__name__)

AgentId = str


class HierarchicalEnv(MultiAgentEnv, ABC, Generic[EnvConfig, EnvState]):
    def __init__(self, config: EnvConfig, agent_configs: Dict[AgentName, Any]):
        self._config = config
        self._agent_configs = agent_configs

        self._validate_transitions_on_done()

        self._agents = self._init_agents()
        # This dict holds the reference how many times a given agent was done, so
        # it can be used to provide a unique id for each new agent of the same type.
        self._agent_counter: Dict[AgentName, int] = defaultdict(int)
        self._current_agent_name: Optional[AgentName] = None
        self._current_agent_id: Optional[AgentId] = None

        self._prev_state: Optional[EnvState] = None

    @cached_property
    @abstractmethod
    def agents(
        self,
    ) -> Dict[
        AgentName, Type[Agent[EnvConfig, EnvState, Any, Any, Any, Any, Action, Any]]
    ]:
        pass

    @cached_property
    def procedures(self) -> Dict[ProcedureName, Procedure[EnvState, ProcedureRequest]]:
        return {}

    @property
    @abstractmethod
    def initial_agent(self) -> AgentName:
        pass

    @cached_property
    def transitions_on_done(self) -> Dict[AgentName, Optional[AgentName]]:
        """
        Defines which agent to use, when one agent (represented by the key in this dict)
        dones. The environment dones, if the value is None.
        There should be at least one entry with None value.
        """
        return {self.initial_agent: None}

    @cached_property
    def transitions_on_action(self) -> List[Tuple[AgentTrigger[Action], AgentName]]:
        """
        Defines triggers, which allow changing the current agent when it outputs
        a specific action. The new agent will receive this action.

        Examples:
            >>> HIGH_LEVEL_AGENT = "high_level"
            ... LOW_LEVEL_AGENT = "low_level"
            ... GO_RIGHT = 2
            ... [(lambda name, action: (
            ...     name == HIGH_LEVEL_AGENT and action == GO_RIGHT
            ... ), LOW_LEVEL_AGENT)]
        """
        return []

    @cached_property
    def procedures_on_action(self) -> List[Tuple[AgentTrigger[Action], ProcedureName]]:
        """ """
        return []

    @abstractmethod
    def initial_state(self) -> EnvState:
        pass

    @abstractmethod
    def env_step(self, state: EnvState, action: Action) -> EnvState:
        pass

    @property
    def observation_space(self) -> Space:
        try:
            agent = self._current_agent
        except KeyError:
            agent = self._agents[self.initial_agent]
        agent_config = self._agent_configs[agent.NAME]
        return agent.observation_space(agent_config, self._config)

    @property
    def action_space(self) -> Space:
        try:
            agent = self._current_agent
        except KeyError:
            agent = self._agents[self.initial_agent]
        agent_config = self._agent_configs[agent.NAME]
        return agent.action_space(agent_config, self._config)

    def reset(self) -> MultiAgentDict:
        self._agent_counter = defaultdict(int)

        for agent in self._agents.values():
            agent.on_reset()

        state = self._prev_state = self.initial_state()
        state = self._switch_agent(self.initial_agent, state)

        obs = {
            self._current_agent_id: self._current_agent.encode_observation(
                self._current_agent.translate_state(state)
            )
        }
        return obs

    def step(
        self, action_dict: MultiAgentDict
    ) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        assert (
            self._prev_state
        ), "The episode is not initialized. Did you forget to call `reset` first?"
        assert self._current_agent, "There should be a single current agent set."
        assert len(action_dict) == 1, (
            "This environment follows a hierarchical reinforcement learning approach "
            "and always expects an action for only one agent."
        )
        assert (
            self._current_agent_id in action_dict
        ), f"Expected an action for agent `{self._current_agent_name}`."

        agent_action = action_dict[self._current_agent_id]
        action = self._current_agent.decode_action(
            self._current_agent.translate_state(self._prev_state), agent_action
        )
        self._current_agent.on_step(action)

        if isinstance(action, SwitchAgent):
            next_agent = self._get_next_agent(action)
            state = self._switch_agent(next_agent, self._prev_state, action)
        elif isinstance(action, ProcedureRequest):
            procedure = self._get_procedure(action)
            state = procedure.execute(self._prev_state, action)
        else:
            state = self.env_step(self._prev_state, action)

        result: Tuple[
            MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
        ] = ({}, {}, {}, {})
        _, _, done, _ = result
        self._populate_result_with_agent_output(result, state, action)

        agent_done = done[self._current_agent_id]

        if agent_done:
            LOG.debug(f"Agent `{self._current_agent_name}` has done.")
            next_agent = self.transitions_on_done[
                self._current_agent_name  # type: ignore
            ]
            if next_agent is not None:
                self._agent_counter[self._current_agent_name] += 1  # type: ignore
                state = self._switch_agent(next_agent, state)
                self._populate_result_with_agent_output(result, state, action)

        done["__all__"] = all(done.values())
        self._prev_state = state

        return result

    @property
    def _current_agent(
        self,
    ) -> Agent[EnvConfig, EnvState, Any, Any, Any, Any, Action, Any]:
        return self._agents[self._current_agent_name]  # type: ignore

    def _validate_transitions_on_done(self) -> None:
        assert any(action is None for action in self.transitions_on_done.values())
        assert self.transitions_on_done.keys() == self.agents.keys()

    def _init_agents(
        self,
    ) -> Dict[AgentName, Agent[EnvConfig, EnvState, Any, Any, Any, Any, Action, Any]]:
        agents = {}
        for agent_name, agent_cls in self.agents.items():
            agent_config = self._agent_configs[agent_name]
            agents[agent_name] = agent_cls(agent_config, self._config)
        return agents

    def _switch_agent(
        self,
        new_agent: AgentName,
        state: EnvState,
        action: Optional[SwitchAgent] = None,
    ) -> EnvState:
        LOG.debug(
            f"Switching the agent from `{self._current_agent_name}` to `{new_agent}`."
        )
        try:
            self._current_agent.on_gives_control(action)
        except KeyError:
            # It's ok, we call `_switch_agent` on reset, so no current agent set yet.
            pass
        self._current_agent_name = new_agent
        self._current_agent_id = self._agent_id(new_agent)
        new_state = self._current_agent.on_takes_control(state, action)
        return new_state

    def _get_next_agent(self, action: SwitchAgent) -> AgentName:
        for trigger, agent in self.transitions_on_action:
            if trigger(self._current_agent_name, action):  # type: ignore
                return agent
        raise MissingNextAgent(self._current_agent, action)

    def _get_procedure(self, action: ProcedureRequest) -> Procedure[EnvState, Any]:
        for trigger, procedure in self.procedures_on_action:
            if trigger(self._current_agent_name, action):  # type: ignore
                return self.procedures[procedure]
        raise MissingProcedure(self._current_agent, action)

    def _agent_id(self, name: AgentName) -> AgentId:
        return f"{name}_{self._agent_counter[name]}"

    def _populate_result_with_agent_output(
        self,
        result: Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict],
        state: EnvState,
        action: Action,
    ) -> None:
        agent_state = self._current_agent.translate_state(state)
        agent_prev_state = self._current_agent.translate_state(
            self._prev_state  # type: ignore
        )

        obs, reward, done, info = result
        obs[self._current_agent_id] = self._current_agent.encode_observation(
            agent_state
        )
        reward[self._current_agent_id] = self._current_agent.calculate_reward(
            agent_prev_state, action, agent_state
        )
        done[self._current_agent_id] = self._current_agent.has_done(agent_state)
        info[self._current_agent_id] = self._current_agent.info(
            agent_prev_state, action, agent_state
        )
