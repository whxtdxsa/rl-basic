hps = {
    'episodes': 100000 
}

episodes = hps['episodes']

from environment import GridWorld
from agent import Disa
env = GridWorld()
agent = Disa() 

from utils import one_hot
from tqdm import tqdm
loss_history = []
for episode in tqdm(range(episodes), desc=f'Episode'):
    state = env.reset()
    state = one_hot(state)
    total_loss, cnt = 0, 0
    done = False

    while not done:
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(state, action)
        next_state = one_hot(next_state)


        loss = agent.update(state, action, reward, next_state, done)
        total_loss += loss
        cnt += 1
        state = next_state

    average_loss = total_loss / cnt
    loss_history.append(average_loss)

# [그림 7-14] 에피소드별 손실 추이
import matplotlib.pyplot as plt
plt.xlabel('episode')
plt.ylabel('loss')
plt.plot(range(len(loss_history)), loss_history)
plt.savefig('q.png')

# [그림 7-15] 신경망을 이용한 Q 러닝으로 얻은 Q 함수와 정책
Q = {}
for state in env.states():
    for action in env.action_space:
        q = agent.Q(one_hot(state))[:, action]
        Q[state, action] = float(q.data)
env.render_q(Q)

