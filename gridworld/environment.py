import numpy as np
from utils import Renderer
class GridWorld:
    def __init__(self):
        
        self.reward_map = np.array([
            [-9, 3, 0, 1],
            [0, None, 0, -9],
            [0, 0, 0, 0]
        ])

        self.action_space = [0, 1, 2, 3]
        self.wall_state = (1, 1)
        self.end_state = (0, 1)
        self.start_state = (2, 0)
        self.agent_state = self.start_state
    @property
    def height(self):
        return len(self.reward_map)

    @property
    def width(self):
        return len(self.reward_map[0])

    @property
    def shape(self):
        return self.reward_map.shape

    def actions(self):
        return self.action_space

    def states(self):
        for h in range(self.height):
            for w in range(self.width):
                yield (h, w)

    def reset(self):
        self.agent_state = self.start_state
        return self.agent_state

    
    def step(self, state, action):
        move_space = [(-1, 0), (1, 0), (0,  -1), (0, 1)]
        move = move_space[action]

        next_state = (move[0] + state[0], move[1] + state[1])
        ny, nx = next_state
    
        done = False
        if self.wall_state == next_state or ny < 0 or ny >= len(self.reward_map) or nx < 0 or nx >= len(self.reward_map[0]):
            next_state = state
            reward = -0.5
        else:
            reward = self.reward_map[ny][nx]

            if next_state == self.end_state:
                done = True
        
        return next_state, reward, done, {}

        
    def render(self):
        print(self.start_state)
    
    def render_v(self, v=None, policy=None, print_value=True):
        renderer = Renderer(self.reward_map, self.end_state,
                                          self.wall_state)
        renderer.render_v(v, policy, print_value)

    def render_q(self, q=None, print_value=True):
        renderer = Renderer(self.reward_map, self.end_state,
                                          self.wall_state)
        renderer.render_q(q, print_value)
    


