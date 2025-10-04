import os
import gymnasium as gym
from ac_agent import ActorCritic
from tqdm import tqdm
from gymnasium.wrappers import RecordVideo

hp = {"num_of_sample": 1, "episodes": 1250, "lr_v": 0.0002, "lr_pi": 0.0002}

f_name = f"{hp['episodes']}_{hp['lr_v']}_{hp['lr_pi']}"
f_dir = "/data/data/com.termux/files/home/storage/downloads/termux/"

env = gym.make("CartPole-v0", render_mode="rgb_array")
env = RecordVideo(env, video_folder=f_dir, episode_trigger=lambda x: True)

log_reward = [0] * hp["episodes"]
for i in range(hp["num_of_sample"]):
    agent = ActorCritic(hp["lr_v"], hp["lr_pi"])
    for episode in tqdm(range(hp["episodes"]), desc=f"{i + 1}/{hp['num_of_sample']}"):
        obs, _ = env.reset()
        done = False

        total_reward = 0
        while not done:
            action, prob = agent.get_action(obs)
            next_obs, reward, terminated, trucated, _ = env.step(action)
            done = terminated | trucated

            agent.update(obs, reward, next_obs, prob, done)

            obs = next_obs
            total_reward += float(reward)

        log_reward[episode] += total_reward / hp["num_of_sample"]

env.close()

import matplotlib.pyplot as plt

plt.plot(log_reward)
plt.savefig(os.path.join(f_dir, f_name + ".png"))
