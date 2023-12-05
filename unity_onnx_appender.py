import os
import numpy as np
from typing import List

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.env import Unity3DEnv

import torch
from torch.nn import Parameter

# Load the ONNX model
# model = onnx.load("./models_soccer/model_goalie.onnx")
# graph = model.graph
# onnx.save(model, "./models_soccer/model_modified_goalie.onnx")

tune.register_env(
    "unity3d",
    lambda c: Unity3DEnv(
        file_name="builds/soccer.app",
        no_graphics=False,
        episode_horizon=c["episode_horizon"],
    ),
)

policies, policy_mapping_fn = Unity3DEnv.get_policy_configs_for_game("SoccerStrikersVsGoalie")
config = (
    PPOConfig()
    .environment(
        "unity3d",
        env_config={
            "file_name": "soccer.app",
            "episode_horizon": 3000,
        },
    )
    .framework("torch")
    .rollouts(
        num_rollout_workers=1,
        rollout_fragment_length=200,
    )
    .training(
        lr=0.0003,
        lambda_=0.95,
        gamma=0.99,
        sgd_minibatch_size=256,
        train_batch_size=4000,
        num_sgd_iter=20,
        clip_param=0.2,
        model={"fcnet_hiddens": [512, 512]},
    )
    .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    .rl_module(_enable_rl_module_api=False)
    .training(_enable_learner_api=False)
)
trainer = PPO(config=config)

# Restore the trainer from the checkpoint
checkpoint_path = ("/Users/wehrenberger/ray_results/PPO_2023-12-01_18-00-08/PPO_unity3d_15170_00000_0_2023-12-01_18-00"
                   "-08/checkpoint_000018")
trainer.restore(checkpoint_path)

# Get the policy
policy = trainer.get_policy("Goalie")  # Striker Goalie
torch_model = policy.model

print(torch_model)
print(policy.action_space)  # G: MultiDiscrete([3 3 3]) S: MultiDiscrete([3 3 3])
print(policy.observation_space)  # G: Box(-inf, inf, (738,), float32) S: Box(-1.0, 1.0, (294,), float32)

# G
example_obs_0 = torch.rand(1, 738)

# S
# example_obs_0 = torch.rand(1, 231)  # Shape matching Unity's obs_0
# example_obs_1 = torch.rand(1, 63)  # Shape matching Unity's obs_1

num_actions = 9
example_actions = torch.randint(low=0, high=3, size=(1, 3))  # Batch size is 1, action size is 3
example_mask = torch.ones(1, num_actions)
example_action_output_shape = torch.Tensor([[3, 3, 3]])

ray.shutdown()


class WrapperNet(torch.nn.Module):
    def __init__(
            self,
            model,
            discrete_output_sizes: List[int],
    ):
        """
        Wraps the VisualQNetwork adding extra constants and dummy mask inputs
        required by runtime inference with Sentis.

        For environment continuous actions outputs would need to add them
        similarly to how discrete action outputs work, both in the wrapper
        and in the ONNX output_names / dynamic_axes.
        """
        super(WrapperNet, self).__init__()
        self.model = model

        # version_number
        #   MLAgents1_0 = 2   (not covered by this example)
        #   MLAgents2_0 = 3
        version_number = torch.Tensor([3])
        self.version_number = Parameter(version_number, requires_grad=False)

        # memory_size
        memory_size = torch.Tensor([0])
        self.memory_size = Parameter(memory_size, requires_grad=False)

        # discrete_action_output_shape
        self.discrete_shape = Parameter(example_action_output_shape, requires_grad=False)

    # def forward(self, obs_0: torch.tensor, obs_1: torch.tensor, mask: torch.tensor):
    def forward(self, obs_0: torch.tensor, mask: torch.tensor):
        # combined_obs = torch.cat([obs_0, obs_1], dim=1)
        input_dict = {
            "obs": obs_0,  # combined_obs
            "action_mask": mask
        }

        model_result = self.model(input_dict)[0]

        # Split the model_result into branches and apply argmax to each branch
        # Assuming model_result is a tensor of shape [batch_size, 9] and you have 3 branches
        branch_logits = torch.split(model_result, 3, dim=1)  # Split into 3 branches
        actions = [torch.argmax(branch, dim=1, keepdim=True) for branch in branch_logits]

        # Concatenate the actions from each branch
        discrete_actions = torch.cat(actions, dim=1)

        return [discrete_actions], self.discrete_shape, self.version_number, self.memory_size


torch.onnx.export(
    WrapperNet(torch_model, [num_actions]),

    # A tuple with an example of the input tensors, For forward?
    (example_obs_0, example_mask),  # , example_obs_1, example_mask),
    'models_soccer/model_goalie_modified.onnx',  # goalie, striker
    opset_version=9,

    # input_names must correspond to the WrapperNet forward parameters
    # obs will be obs_0, obs_1, etc.
    input_names=["obs_0", "obs_1", "action_masks"],

    # output_names must correspond to the return tuple of the WrapperNet
    # forward function.
    output_names=["discrete_actions", "discrete_action_output_shape",
                  "version_number", "memory_size"],
    # All inputs and outputs should have their 0th dimension be designated
    # as 'batch'
    dynamic_axes={'obs_0': {0: 'batch'},  # 'obs_1': {0: 'batch'},
                  'action_masks': {0: 'batch'},
                  'discrete_actions': {0: 'batch'},
                  'discrete_action_output_shape': {0: 'batch'}
                  }
)
