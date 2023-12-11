import os

import numpy as np

from gymnasium.spaces import Box, Tuple as TupleSpace

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.unity3d_env import Unity3DEnv
from ray.rllib.policy.policy import PolicySpec

# ToDo
# - calc right batch size

print("HIIIIIIII 0")

ray.shutdown()

print("HIIIIIIII 1")

# object_store_memory=20 * 1024 * 1024 * 1024
ray.init()
# ray.init(address='192.168.193.207:6379')
# ray.init(address='auto')

print("HIIIIIIII 2")

ray.util.get_node_ip_address()

print("HIIIIIIII 3")

game_name = "AIBattleSim"
agent = "FPS_Agent"

env = "CentOS_FPSAgent_Env1_C2"
file_name = f"/home/lu72hip/mlagents_rllib_bridge/builds/FPSAgent/{env}/env.x86_64"  # None for editor
checkpoint_dir = f"/home/lu72hip/mlagents_rllib_bridge/checkpoints/{game_name}"

# CONFIGURE BATCH
episode_horizon = 3000
stop_time_steps = 10_000_000

NUM_ROLLOUT_WORKERS = 1
NUM_GPUS = 0

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
        # num_envs_per_worker=NUM_ENVS_PER_WORKER,
        rollout_fragment_length="auto"
        # ignore_worker_failures=True
    )
    # .resources(
    #     custom_resources_per_worker={"worker": 1}
    # )
    .training(
        lr=0.0003,
        lambda_=0.95,
        gamma=0.99,
        sgd_minibatch_size=128,
        train_batch_size=512, # THIS ONE
        num_sgd_iter=10,
        clip_param=0.2,
        model={
            "conv_filters": [
                [16, [2, 2], 1],  # 16 filters, 3x3 kernel, stride 1
                # [32, [2, 2], 1],  # 32 filters, 3x3 kernel, stride 1
                # [64, [2, 2], 2],  # 64 filters, 3x3 kernel, stride 2
            ],
            "fcnet_hiddens": [256, 256], # 1024, 1024],
            # "lstm_cell_size": 1024,
            # "max_seq_len": 256,
            # "use_attention": True,  # Note: This option is not standard in RLlib
            # "custom_model": "custom_cnn_v1"
            # "_disable_preprocessor_api": True
        },
    )
    .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    .resources(num_gpus=NUM_GPUS) #int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    .rl_module(_enable_rl_module_api=False)
    .training(_enable_learner_api=False)
)

stop = {
    "training_iteration": 999_999_999,
    "timesteps_total": stop_time_steps,
    "episode_reward_mean": 999_999_999,
}

# Run the experiment.
results = tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=air.RunConfig(
        stop=stop,
        verbose=1,
        local_dir=checkpoint_dir,
        checkpoint_config=air.CheckpointConfig(
            checkpoint_frequency=100,
            checkpoint_at_end=True,
            num_to_keep=10,
        ),
    )
).fit()

ray.shutdown()
