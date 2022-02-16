from typing import Any, List, Optional

import pytest
from maze.agent.motion import MotionAgent
from maze.agent.strategy import StrategyAgent
from maze.env import MazeEnv
from maze.env_config import DEFAULTS
from maze.exceptions import DirectionNonWalkable
from maze.maze import Direction
from ray.rllib.utils.typing import MultiAgentDict

from hrl.env import HierarchicalEnv


@pytest.fixture
def env() -> MazeEnv:
    common_config = DEFAULTS | {}
    return MazeEnv(
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


def test_maze_env_strategy_directions_mask(env: MazeEnv) -> None:
    obs = env.reset()
    assert obs["strategy_0"]["directions_mask"].tolist() == [1.0, 0.0, 0.0, 1.0]


def test_maze_env_strategy_raises_on_invalid_direction(env: MazeEnv) -> None:
    env.reset()
    try:
        env.step({"strategy_0": Direction.RIGHT.value})
        pytest.fail()
    except DirectionNonWalkable:
        pass


def test_maze_env_motion_directions_mask(env: MazeEnv) -> None:
    env.reset()
    obs, _, _, _ = env.step({"strategy_0": Direction.UP.value})
    assert obs["motion_0"]["directions_mask"].tolist() == [0.0, 1.0]

    obs, _, _, _ = env.step({"motion_0": 1})
    assert obs["motion_0"]["directions_mask"].tolist() == [1.0, 0.0]


def test_maze_env_motion_raises_on_invalid_direction(env: MazeEnv) -> None:
    env.reset()
    env.step({"strategy_0": Direction.UP.value})
    try:
        env.step({"motion_0": 0})
        pytest.fail()
    except DirectionNonWalkable:
        pass


def test_maze_env_strategy_times_out(env: MazeEnv) -> None:
    env.reset()
    direction = Direction.DOWN
    done = {"__all__": False}
    for i in range(10):
        assert not done["__all__"]
        direction = Direction.opposite(direction)
        _, _, done, _ = env.step({"strategy_0": direction.value})
        assert not done["__all__"]
        _, _, done, _ = env.step({f"motion_{i}": 1})
    assert done["__all__"]


def test_maze_env_motion_times_out(env: MazeEnv) -> None:
    env.reset()
    env.step({"strategy_0": Direction.LEFT.value})
    env.step({"motion_0": 1})
    env.step({"strategy_0": Direction.DOWN.value})
    env.step({"motion_1": 1})
    env.step({"motion_1": 1})
    env.step({"strategy_0": Direction.LEFT.value})
    env.step({"motion_2": 1})

    result = {}, {}, {}, {}
    for _ in range(4):
        result = env.step({"motion_2": 0})
    assert_agents(["motion_2", "strategy_0"], env, *result)
    _, _, done, _ = result
    assert done["motion_2"]


def test_maze_env_on_successful_path(env: MazeEnv) -> None:
    obs = env.reset()
    assert_agent("strategy_0", env, obs)

    result = env.step({"strategy_0": Direction.LEFT.value})
    assert_agent("motion_0", env, *result)

    result = env.step({"motion_0": 1})
    assert_agents(["motion_0", "strategy_0"], env, *result)
    _, _, done, _ = result
    assert done["motion_0"]

    result = env.step({"strategy_0": Direction.DOWN.value})
    assert_agent("motion_1", env, *result)

    result = env.step({"motion_1": 1})
    assert_agent("motion_1", env, *result)

    result = env.step({"motion_1": 1})
    assert_agents(["motion_1", "strategy_0"], env, *result)
    _, _, done, _ = result
    assert done["motion_1"]

    result = env.step({"strategy_0": Direction.LEFT.value})
    assert_agent("motion_2", env, *result)

    result = env.step({"motion_2": 1})
    assert_agent("motion_2", env, *result)

    result = env.step({"motion_2": 1})
    assert_agent("motion_2", env, *result)

    result = env.step({"motion_2": 1})
    assert_agents(["motion_2", "strategy_0"], env, *result)
    _, _, done, _ = result
    assert done["motion_2"]

    result = env.step({"strategy_0": Direction.UP.value})
    assert_agent("motion_3", env, *result)

    result = env.step({"motion_3": 1})
    assert_agent("motion_3", env, *result)

    result = env.step({"motion_3": 1})
    assert_agents(["motion_3", "strategy_0"], env, *result)
    _, _, done, _ = result
    assert done["motion_3"]

    result = env.step({"strategy_0": Direction.UP.value})
    assert_agent("motion_4", env, *result)

    result = env.step({"motion_4": 1})
    assert_agent("motion_4", env, *result)

    result = env.step({"motion_4": 1})
    assert_agent("motion_4", env, *result)

    result = env.step({"motion_4": 1})
    assert_agents(["motion_4", "strategy_0"], env, *result)
    _, _, done, _ = result
    assert done["motion_4"]

    result = env.step({"strategy_0": Direction.LEFT.value})
    assert_agent("motion_5", env, *result)

    result = env.step({"motion_5": 1})
    assert_agents(["motion_5", "strategy_0"], env, *result)
    _, _, done, _ = result
    assert done["motion_5"]

    result = env.step({"strategy_0": Direction.UP.value})
    assert_agent("motion_6", env, *result)

    result = env.step({"motion_6": 1})
    assert_agents(["motion_6", "strategy_0"], env, *result)
    _, _, done, _ = result
    assert done["motion_6"]
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
        assert len(info) == num_agents
        for agent in expected:
            assert agent in info
            assert isinstance(info[agent], dict)
