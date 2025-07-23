import gymnasium as gym
import sys
print(sys.executable)
import metaworld

env = gym.make("Meta-World/MT1", env_name="reach-v3",render_mode="human")

observation, info = env.reset()
for _ in range(500):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()


env.close()