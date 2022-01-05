from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import override
from ray.rllib.utils.typing import TensorType


class MazeModel(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **custom_model_config
    ):
        nn.Module.__init__(self)
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        self._custom_model_config = custom_model_config

        map = self._custom_model_config["map"]
        rows, cols = len(map), len(map[0])
        self._map_size = rows * cols
        position_size = 2
        input_features = self._map_size + position_size

        num_actions = self._custom_model_config["num_actions"]

        self.fc1 = nn.Linear(input_features, 128)
        self.fc2 = nn.Linear(128, 128)
        self.policy_head = nn.Linear(128, num_actions)
        self.value_head = nn.Linear(128, 1)

        self._last_value: Optional[torch.Tensor] = None

    @override(ModelV2)
    def get_initial_state(self) -> List[np.ndarray]:
        return []

    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        obs = input_dict["obs"]
        map_flatten = torch.reshape(obs["map"], [-1, self._map_size])
        features = torch.cat((map_flatten, obs["position"]), dim=-1)

        x = F.relu(self.fc1(features))
        x = F.relu(self.fc2(x))

        self._last_value = self.value_head(x)

        policy_logits = self.policy_head(x)
        mask = obs["directions_mask"].bool()
        filtered_policy_logits = torch.where(
            mask,
            policy_logits,
            torch.tensor(-torch.finfo(torch.float32).max, device=policy_logits.device),
        )

        return filtered_policy_logits, state

    def value_function(self) -> TensorType:
        return torch.reshape(self._last_value, [-1])
