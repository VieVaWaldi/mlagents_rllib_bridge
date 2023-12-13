import os

import numpy as np

from gymnasium.spaces import Box, Tuple as TupleSpace

import ray
from ray import air, tune
from ray.rllib.algorithms.ddppo import DDPPOConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.unity3d_env import Unity3DEnv
from ray.rllib.policy.policy import PolicySpec

# ToDo
# - calc right batch size

# ray.init()
ray.init(address='192.168.194.13:6379', include_dashboard=True, dashboard_host='0.0.0.0', dashboard_port=8265)

game_name = "AIBattleSim"
agent = "FPS_Agent"

env = "CentOS_FPSAgent_Env1_C2"
file_name = f"/home/lu72hip/mlagents_rllib_bridge/builds/FPSAgent/{env}/env.x86_64"  # None for editor
checkpoint_dir = f"/home/lu72hip/mlagents_rllib_bridge/checkpoints/{game_name}"

# CONFIGURE BATCH
episode_horizon = 3000
stop_time_steps = 10_000_000

NUM_ROLLOUT_WORKERS = 22 # probably globally
NUM_GPUS = 4 # probably per worker

tune.register_env(
    game_name,
    lambda c: Unity3DEnv(
        file_name=file_name,
        no_graphics=True,
        episode_horizon=episode_horizon,
    ),
)

obs_spaces = TupleSpace([
    Box(low=0, high=1, shape=(27, 40, 2), dtype=np.float32),
    Box(low=0, high=1, shape=(27, 40, 2), dtype=np.float32),
    Box(low=0, high=1, shape=(27, 40, 6), dtype=np.float32),
    Box(low=0, high=1, shape=(27, 40, 2), dtype=np.float32),
    Box(low=0, high=1, shape=(31,), dtype=np.float32),
])

action_spaces = Box(-1.0, 1.0, (5,), dtype=np.float32)

# SHARED POLICY AGENT SETUP ###
policies = {
    agent: PolicySpec(
        observation_space=obs_spaces,
        action_space=action_spaces,
    ),
}


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return agent


config = (
    PPOConfig()
    # DDPPOConfig()
    .environment(
        game_name,
        env_config={
            "file_name": file_name,
            "episode_horizon": episode_horizon,
        },
    )
    .framework("torch")
    .rollouts(
        num_rollout_workers=NUM_ROLLOUT_WORKERS,
        rollout_fragment_length="auto",
        ignore_worker_failures=True,
        recreate_failed_workers=True
    )
    .resources(
        num_gpus=NUM_GPUS,
        placement_strategy="SPREAD",
        
        num_cpus_per_worker=20, # Each rollout worker gets 1 CPU
        num_gpus_per_worker=0,  # Rollout workers do not use GPU

        num_learner_workers=2,  # Number of workers for training
        num_cpus_per_learner_worker=1,  # Each training worker gets 1 CPU
        num_gpus_per_learner_worker=1,  # Each training worker gets 1 GPU

        # custom_resources_per_worker={"gpu_node": 1}  # If using custom resources
    )
    .rl_module(_enable_rl_module_api=True)
    .training(
        _enable_learner_api=True,
        lr=0.0003,
        lambda_=0.95,
        gamma=0.99,
        sgd_minibatch_size=1024,
        train_batch_size=12288, # THIS ONE
        num_sgd_iter=10,
        clip_param=0.2,
        model={
            "conv_filters": [
                [16, [2, 2], 1],  # 16 filters, 3x3 kernel, stride 1
                [32, [2, 2], 1],  # 32 filters, 3x3 kernel, stride 1
                # [64, [2, 2], 2],  # 64 filters, 3x3 kernel, stride 2
            ],
            # "fcnet_hiddens": [256, 256]
            "fcnet_hiddens": [1024, 1024, 1024, 1024],
            # "lstm_cell_size": 1024,
            # "max_seq_len": 256,
            # "use_attention": True,  # Note: This option is not standard in RLlib
            # "custom_model": "custom_cnn_v1"
            # "_disable_preprocessor_api": True
        },
    )
    .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
)

stop = {
    "training_iteration": 999_999_999,
    "timesteps_total": stop_time_steps,
    "episode_reward_mean": 999_999_999,
}

# Run the experiment.
# results = tune.Tuner(
#     "PPO",
#     param_space=config.to_dict(),
#     run_config=air.RunConfig( 
#         stop=stop,
#         verbose=1,
#         local_dir=checkpoint_dir,
#         checkpoint_config=air.CheckpointConfig(
#             checkpoint_frequency=100,
#             checkpoint_at_end=True,
#             num_to_keep=10,
#         ),
#     )
# ).fit()

from ray.rllib.algorithms.algorithm import Algorithm
algo = Algorithm.from_checkpoint("/home/lu72hip/mlagents_rllib_bridge/checkpoints/AIBattleSim/PPO_2023-12-12_21-22-28/PPO_AIBattleSim_2c3f1_00000_0_2023-12-12_21-22-29/checkpoint_000002")

for i in range(1000):
    for i in range(100):
        result = algo.train()
        print(result)
    new_checkpoint = algo.save("/home/lu72hip/mlagents_rllib_bridge/checkpoints/custom/AIBattleSim/from_PPO_AIBattleSim_2c3f1_00000_0_2023-12-12_21-22-29_checkpoint_000002")



ray.shutdown()
