import gymnasium as gym
import matplotlib.pyplot as plt
from agent import DQNAgent, RandomAgent
from tqdm import tqdm


hp = {
    "sync_interval": 5,
    "episodes": 200,
    "buffer_size": 10000,
    "batch_size": 64,
    "learning_rate": 0.0003,
}

env = gym.make("CartPole-v0")
reward_history = [0] * hp["episodes"]

for i in range(10):
    agent = DQNAgent(hp["buffer_size"], hp["batch_size"], hp["learning_rate"])
    for episode in tqdm(range(hp["episodes"]), desc=f"{i + 1}/10"):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminate, truncate, _ = env.step(action)
            done = terminate | truncate
            agent.update(obs, action, reward, next_obs, done)

            obs = next_obs
            total_reward += float(reward)

        if episode % hp["sync_interval"] == 0:
            agent.sync_qnet()

        reward_history[episode] += total_reward / 10


f_name = f"{hp['sync_interval']}_{hp['buffer_size']}_{hp['batch_size']}_{hp['learning_rate']}.png"
plt.plot(reward_history)
plt.savefig(f"/data/data/com.termux/files/home/storage/downloads/termux/{f_name}")
