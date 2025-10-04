from tictactoc_world import TicTacTocWorld
from q_learning import QLearningAgent
from utils import load_q_table

hps = {
    'load_table': True,
}

env = TicTacTocWorld()
agent = QLearningAgent()
agent.epsilon = 0.0

if hps['load_table']:
    load_q_table(agent)

for round in range(2):
    print('-----------------------')
    print(f'Round {round + 1} start')
    print('-----------------------')
    state = env.reset()
    if round == 1: env.curr_player = 1
    else: env.curr_player = -1
    if env.curr_player == 1:
        agent_action = agent.get_action(state)    
        state, reward, done, _ = env.step(agent_action)
        print(f'Agent chooses: {agent_action}')

    env.render()


    while True:
        valid_actions = [i for i, s in enumerate(state) if s == 0]
        action = int(input(f"select action {valid_actions}: "))
        while action not in valid_actions:
            action = int(input(f"Invalid! Choose again {valid_actions}: "))
        state, reward, done, _ = env.step(action)

        env.render()
        if done:
            print('Game Over!')
            if reward == 1:
                print('Player win!')
            else:
                print('Draw')
            break

        agent_action = agent.get_action(state)    
        state, reward, done, _ = env.step(agent_action)
        print(f'Agent chooses: {agent_action}')

        env.render()
        if done:
            print('Game Over!')
            if reward == 1:
                print('Agent win!')
            else:
                print('Draw')
            break
