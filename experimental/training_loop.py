
# env = gym.make(game_name, env_config={
#     "file_name": file_name,
#     "episode_horizon": episode_horizon
# })

# agent = PPO(config=config)

# Training Loop
# max_episodes = 1000
# max_steps = 200  # Assuming this is your desired number of steps per episode

# for episode in range(max_episodes):
#     obs = env.reset()
#     total_reward = 0
#     for step in range(max_steps):
#         # Generate an action from the policy
#         action = agent.compute_single_action(obs)
#
#         # Step the environment
#         obs, reward, done, info = env.step(action)
#         total_reward += reward
#
#         # Handle end of episode
#         if done:
#             print(f"Episode {episode} finished after {step+1} steps with reward {total_reward}.")
#             break
#
#     # Optional: Perform a training step after each episode
#     agent.train()
