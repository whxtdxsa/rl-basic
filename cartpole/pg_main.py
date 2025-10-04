import gymnasium as gym
import matplotlib.pyplot as plt
from agent import PGAgent, REINFORCE
from tqdm import tqdm

algorithm = "reinforce"
hp = {
    "sync_interval": 5,
    "episodes": 3000,
    "buffer_size": 10000,
    "batch_size": 1,
    "learning_rate": 0.0003,
    "num_of_sample": 20,
}

env = gym.make("CartPole-v1")
reward_history = [0] * hp["episodes"]

for i in range(hp["num_of_sample"]):
    agent = REINFORCE(hp["buffer_size"], hp["batch_size"], hp["learning_rate"])
    for episode in tqdm(range(hp["episodes"]), desc=f"{i + 1}/{hp['num_of_sample']}"):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, prob = agent.get_action(obs)
            next_obs, reward, terminate, truncate, _ = env.step(action)
            done = terminate | truncate
            agent.add(prob, reward)

            obs = next_obs
            total_reward += float(reward)

        agent.update()
        reward_history[episode] += total_reward / hp["num_of_sample"]


f_name = f"{algorithm}_{hp['sync_interval']}_{hp['episodes']}_{hp['buffer_size']}_{hp['batch_size']}_{hp['learning_rate']}.png"
plt.plot(reward_history)
plt.savefig(f"/data/data/com.termux/files/home/storage/downloads/termux/{f_name}")
