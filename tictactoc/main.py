from tictactoc_world import TicTacTocWorld
from q_learning import QLearningAgent, RandomAgent
from tqdm import tqdm
from utils import save_q_table, load_q_table
import copy

hps = {
    'load_table': False,
    'save_table': True
}


env = TicTacTocWorld()
agent = QLearningAgent()
opponent = QLearningAgent() 


if hps['load_table']:
    load_q_table(agent)

def agent_perspective(state, player):
    return tuple([s * player for s in state])

episodes = 100000
win_rates = []
draw_rates = []
loss_rates = []

wins, draws, losses = 0, 0, 0

opponent.Q = agent.Q
opponent.epsilon = 0.0

for episode in tqdm(range(episodes)):
    state = env.reset()
    done = False
    last_experience = None

    agent_is_player1 = (env.curr_player == 1)
    

    while not done:
        current_player_sign = env.curr_player
        current_player = agent if current_player_sign == 1 else opponent

        perspective_state = agent_perspective(state, env.curr_player)

        action = current_player.get_action(perspective_state)
        next_state, reward, done, _ = env.step(action)

        if last_experience is not None:
            prev_player, prev_state, prev_action, prev_player_sign = last_experience
            prev_player.update(prev_state, prev_action, reward * -1, agent_perspective(next_state, prev_player_sign), done)

        if done:
            current_player.update(perspective_state, action, reward, agent_perspective(next_state, env.curr_player), done)

            if reward == 1:
                if current_player == agent:
                    wins += 1
                else:
                    losses += 1

            else:
                draws += 1
        else:
            last_experience = (current_player, perspective_state, action, current_player_sign)
            state = next_state

    if (episode + 1) % 1000 == 0:
        win_rates.append(wins / 100)
        draw_rates.append(draws / 100)
        loss_rates.append(losses / 100)
        wins, draws, losses = 0, 0, 0


if hps['save_table']:
    save_q_table(agent)

import matplotlib.pyplot as plt       
plt.plot(win_rates, label='Win rate')
plt.plot(draw_rates, label='Draw rate')
plt.plot(loss_rates, label='Loss rate')
plt.xlabel('x100 episodes')
plt.ylabel('rate')
plt.legend()
plt.savefig('graph.png')
env.render()


