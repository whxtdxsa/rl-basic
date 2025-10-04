import random


class TicTacTocWorld:
    def __init__(self):
        self.board = [0] * 9
        self.curr_player = random.choice([-1, 1])

        self.action_space = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.wins = [
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),  # rows
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),  # cols
            (0, 4, 8),
            (2, 4, 6),  # diagonals
        ]

    def reset(self):
        self.board = [0] * 9
        self.curr_player = random.choice([-1, 1])

        return tuple(self.board)

    def step(self, action):
        """
        return: next_state, reward, done, info
        """
        self.board[action] = self.curr_player
        if self.check_win():
            return tuple(self.board), 1, True, {}  # win
        elif 0 not in self.board:
            return tuple(self.board), 0, True, {}  # draw
        self.curr_player *= -1

        return tuple(self.board), 0, False, {}

    def check_win(self):
        for s0, s1, s2 in self.wins:
            if self.board[s0] == self.board[s1] == self.board[s2] != 0:
                return True

        return False

    def switch(self, o):
        if o == 1:
            return " O "
        if o == -1:
            return " X "
        if o == 0:
            return "   "

    def render(self):
        t = list(map(lambda x: self.switch(x), self.board))

        print("|".join(t[0:3]))
        print("-----------")
        print("|".join(t[3:6]))
        print("-----------")
        print("|".join(t[6:9]))
        print()
