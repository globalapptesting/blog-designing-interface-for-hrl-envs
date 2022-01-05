import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from ray.tune.integration.wandb import WandbLoggerCallback

from maze.agent.motion import MotionAgentConfig, MotionAgent
from maze.agent.strategy import StrategyAgentConfig, StrategyAgent
from maze.env import MazeEnv
from maze.env_config import DEFAULTS
from maze.maze import Direction
from maze.model import MazeModel


def register_envs():
    register_env(
        "MazeEnv",
        lambda env_config: MazeEnv(env_config["common"], env_config["agents"]),
    )


def register_models():
    ModelCatalog.register_custom_model("MazeModel", MazeModel)


def train(log_to_wandb: bool):
    common_config = DEFAULTS | {}

    strategy_agent_config: StrategyAgentConfig = StrategyAgent.DEFAULTS | {
        "max_steps": 50,
    }

    strategy_agent_model_config = {
        "custom_model": "MazeModel",
        "custom_model_config": {
            "map": tune.sample_from(lambda spec: spec.config.env_config.common.map),
            "num_actions": len(Direction),
        },
    }

    motion_agent_config: MotionAgentConfig = MotionAgent.DEFAULTS | {
        "max_steps": 20,
    }

    motion_agent_model_config = {
        "custom_model": "MazeModel",
        "custom_model_config": {
            "map": tune.sample_from(lambda spec: spec.config.env_config.common.map),
            "num_actions": 2,
        },
    }

    config = {
        "env": "MazeEnv",
        "env_config": {
            "common": common_config,
            "agents": {
                StrategyAgent.NAME: strategy_agent_config,
                MotionAgent.NAME: motion_agent_config,
            },
        },
        "multiagent": {
            "policies": {
                StrategyAgent.NAME: [
                    None,
                    StrategyAgent.observation_space(
                        strategy_agent_config, common_config
                    ),
                    StrategyAgent.action_space(strategy_agent_config, common_config),
                    {
                        "model": strategy_agent_model_config,
                        "entropy_coeff": 0.5,
                    },
                ],
                MotionAgent.NAME: [
                    None,
                    MotionAgent.observation_space(motion_agent_config, common_config),
                    MotionAgent.action_space(motion_agent_config, common_config),
                    {
                        "model": motion_agent_model_config,
                        "entropy_coeff": 0.01,
                    },
                ],
            },
            "policy_mapping_fn": lambda agent_id: agent_id.split("_")[0],
            "policies_to_train": [StrategyAgent.NAME, MotionAgent.NAME],
            "count_steps_by": "agent_steps",
        },
        "num_workers": 10,
        "num_gpus": 1,
        "framework": "torch",
        "rollout_fragment_length": 10,
        "train_batch_size": 1000,
        "sgd_minibatch_size": 200,
        "num_sgd_iter": 25,
        "grad_clip": 5.0,
        "gamma": 0.999,
        "lambda": 0.9,
        "lr": 0.0001,
        "seed": tune.grid_search([0, 42, 1337]),
    }

    callbacks = []
    if log_to_wandb:
        api_kwargs = {"api_key_file": "~/.wandb"}
        wandb_callback = WandbLoggerCallback(
            entity="tomasz-wrona-gat", project="hrl", **api_kwargs
        )
        callbacks.append(wandb_callback)

    tune.run(
        "PPO",
        config=config,
        callbacks=callbacks,
        stop={
            "timesteps_total": 100_000,
        },
    )


def main():
    ray.init()
    register_envs()
    register_models()
    train(log_to_wandb=False)


if __name__ == "__main__":
    main()
