import os

import numpy as np

from gymnasium.spaces import Box, Tuple as TupleSpace

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.unity3d_env import Unity3DEnv
from ray.rllib.policy.policy import PolicySpec

# from torch_models.custom_cnn_v1 import custom_cnn_v1

ray.init()  # Debug local_mode=True

# ModelCatalog.register_custom_model("custom_cnn_v1", custom_cnn_v1)

game_name = "AIBattleSim"
file_name = None  # "/Users/wehrenberger/Code/AI_BATTLE_SIM/rllib_test/builds/fps_env1_C2.app"  # None for editor
episode_horizon = 3000

agent = "FPS_Agent"
agent_a = "FPS_Agent_A"
agent_b = "FPS_Agent_B"

stop_iters = 9999999
stop_time_steps = 10_000_000
stop_reward = 9999

checkpoint_dir = "/Users/wehrenberger/Code/AI_BATTLE_SIM/rllib_test/checkpoints/battle_sim"
checkpoint_continue_dir = "/Users/wehrenberger/Code/AI_BATTLE_SIM/rllib_test/checkpoints/battle_sim/PPO_2023-12-05_22-10-37/PPO_AIBattleSim_bca0e_00000_0_2023-12-05_22-10-37/checkpoint_005579"

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


# MULTI AGENT SETUP, means we train 2 models ###

# policies = {
#     agent_a: PolicySpec(
#         observation_space=obs_spaces,
#         action_space=action_spaces,
#     ),
#     agent_b: PolicySpec(
#         observation_space=obs_spaces,
#         action_space=action_spaces,
#     ),
# }
#
#
# def policy_mapping_fn(agent_id, episode, worker, **kwargs):
#     return agent_a if "1_" in agent_id else agent_b


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
    # For running in editor, force to use just one Worker
    .rollouts(
        num_rollout_workers=1,
        rollout_fragment_length=200,
        ignore_worker_failures=True,
    )
    .checkpointing(
    )
    .training(
        lr=0.0003,
        lambda_=0.95,
        gamma=0.99,
        sgd_minibatch_size=128,  # 256
        train_batch_size=1000,  # 4000
        num_sgd_iter=10,  # 20
        clip_param=0.2,
        model={
            "conv_filters": [
                [16, [2, 2], 1],  # 16 filters, 3x3 kernel, stride 1
                [32, [2, 2], 1],  # 32 filters, 3x3 kernel, stride 1
                # [64, [2, 2], 2],  # 64 filters, 3x3 kernel, stride 2
            ],
            "fcnet_hiddens": [256, 256],
            # "lstm_cell_size": 256,
            # "max_seq_len": 20,
            # "use_attention": True,  # Note: This option is not standard in RLlib
            # "custom_model": "custom_cnn_v1"
            # "_disable_preprocessor_api": True
        },
    )
    .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    .rl_module(_enable_rl_module_api=False)
    .training(_enable_learner_api=False)
)

stop = {
    "training_iteration": stop_iters,
    "timesteps_total": stop_time_steps,
    "episode_reward_mean": stop_reward,
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
            checkpoint_frequency=500,
            checkpoint_at_end=True,
            num_to_keep=10,
        ),
    )
).fit()

# trainer = config.build()
# trainer.restore(checkpoint_continue_dir)
# trainer.train()

ray.shutdown()
