from typing import Any, List, Optional

import pytest
from maze.agent.motion import MotionAgent
from maze.agent.strategy import StrategyAgent
from maze.env import MazeEnv
from maze.env_config import DEFAULTS
from maze.maze import Direction
from maze_procedure.env import MazeProcedureEnv
from ray.rllib.utils.typing import MultiAgentDict

from hrl.env import HierarchicalEnv


@pytest.fixture
def env() -> MazeProcedureEnv:
    common_config = DEFAULTS | {}
    return MazeProcedureEnv(
        common_config,
        {
            StrategyAgent.NAME: StrategyAgent.DEFAULTS
            | {
                "max_steps": 10,
            },
            MotionAgent.NAME: MotionAgent.DEFAULTS
            | {
                "max_steps": 5,
            },
        },
    )


def test_maze_env_on_successful_path(env: MazeEnv) -> None:
    obs = env.reset()
    assert_agent("strategy_0", env, obs)
    assert obs["strategy_0"]["position"].tolist() == [4, 9]

    result = env.step({"strategy_0": Direction.LEFT.value})
    assert_agent("strategy_0", env, *result)
    obs, *_ = result
    assert obs["strategy_0"]["position"].tolist() == [4, 8]

    result = env.step({"strategy_0": Direction.DOWN.value})
    assert_agent("strategy_0", env, *result)
    obs, *_ = result
    assert obs["strategy_0"]["position"].tolist() == [6, 8]

    result = env.step({"strategy_0": Direction.LEFT.value})
    assert_agent("strategy_0", env, *result)
    obs, *_ = result
    assert obs["strategy_0"]["position"].tolist() == [6, 5]

    result = env.step({"strategy_0": Direction.UP.value})
    assert_agent("strategy_0", env, *result)
    obs, *_ = result
    assert obs["strategy_0"]["position"].tolist() == [4, 5]

    result = env.step({"strategy_0": Direction.UP.value})
    assert_agent("strategy_0", env, *result)
    obs, *_ = result
    assert obs["strategy_0"]["position"].tolist() == [1, 5]

    result = env.step({"strategy_0": Direction.LEFT.value})
    assert_agent("strategy_0", env, *result)
    obs, *_ = result
    assert obs["strategy_0"]["position"].tolist() == [1, 4]

    result = env.step({"strategy_0": Direction.UP.value})
    assert_agent("strategy_0", env, *result)
    obs, *_ = result
    assert obs["strategy_0"]["position"].tolist() == [0, 4]
    _, _, done, _ = result
    assert done["strategy_0"]
    assert done["__all__"]


def assert_agent(
    expected: str,
    env: HierarchicalEnv[Any, Any],
    obs: MultiAgentDict,
    reward: Optional[MultiAgentDict] = None,
    done: Optional[MultiAgentDict] = None,
    info: Optional[MultiAgentDict] = None,
) -> None:
    return assert_agents([expected], env, obs, reward, done, info)


def assert_agents(
    expected: List[str],
    env: HierarchicalEnv[Any, Any],
    obs: MultiAgentDict,
    reward: Optional[MultiAgentDict] = None,
    done: Optional[MultiAgentDict] = None,
    info: Optional[MultiAgentDict] = None,
) -> None:
    num_agents = len(expected)

    assert isinstance(obs, dict)
    assert len(obs) == num_agents
    obs_space_matches = []
    for agent in expected:
        assert agent in obs
        obs_space_match = env.observation_space.contains(obs[agent])
        obs_space_matches.append(obs_space_match)
    assert any(obs_space_matches)

    if reward is not None:
        assert isinstance(reward, dict)
        assert len(reward) == num_agents
        for agent in expected:
            assert agent in reward
            assert isinstance(reward[agent], float)

    if done is not None:
        assert isinstance(done, dict)
        assert len(done) == num_agents + 1
        assert "__all__" in done
        for agent in expected:
            assert agent in done
            assert isinstance(done[agent], bool)

    if info is not None:
        assert isinstance(info, dict)
        assert len(info) == num_agents + 1
        for agent in expected:
            assert agent in info
            assert "__common__" in info
            assert isinstance(info[agent], dict)
