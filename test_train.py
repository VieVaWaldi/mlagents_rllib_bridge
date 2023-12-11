import gym
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO

# Initialize Ray
ray.init()

# Configuration for the training
config = {
    "env": "CartPole-v1",   # Gym environment to use
    "num_workers": 4,       # Number of parallel workers
    "train_batch_size": 400,
    "framework": "tf",      # Specify the deep learning framework (tf or torch)
}

# Training process
results = tune.run(PPOTrainer, config=config, stop={"training_iteration": 100})

# Shut down Ray
ray.shutdown()
