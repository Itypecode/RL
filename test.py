
import numpy as np
from environment.warehouse import WarehouseEnv

env = WarehouseEnv(render_mode="human")

observation, info = env.reset(seed=42)

print("Observation shape:", observation.shape)
print("Action space:", env.action_space)
print("Initial info:", info)

for episode_step in range(1000):

    # Random actions for testing only
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print("Episode ended:", info)
        observation, info = env.reset()

env.close()
