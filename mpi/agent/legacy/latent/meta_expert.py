from metaworld.policies import *
import metaworld
import gymnasium as gym
import random

policy = SawyerReachV3Policy()  # 不同任务使用不同 expert policy

env = gym.make("Meta-World/MT1", env_name="reach-v3",render_mode="human")

observation, info = env.reset()
for _ in range(10000):
    if random.random() > 0.7:
        action = policy.get_action(observation)
    else:
        action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    print(reward)

env.close()