from collections import defaultdict
import numpy as np
def greedy_probs(Q, state, epsilon, action_space):
    qs = [Q[(state, action)] for action in action_space]
    max_q = np.max(qs)
    candidates = [a for a, q in zip(action_space, qs) if q == max_q]
    max_action = np.random.choice(candidates)

    base_prob = epsilon / len(action_space)
    action_probs = {action: base_prob for action in action_space}
    action_probs[max_action] += 1 - epsilon
    return action_probs
        
import json
def save_q_table(agent, file='table.json'):
    t = {str(k): v for k, v in agent.Q.items()}
    with open(file, 'w') as f:
        json.dump(t, f)
    print(f'Q-table saved to {file}')

def load_q_table(agent, file='table.json'):
    with open(file, 'r') as f:
        t = json.load(f)
    agent.Q = defaultdict(lambda: 0, {eval(k): v for k, v in t.items()})
    print(f'Q-table loaded from {file}')
