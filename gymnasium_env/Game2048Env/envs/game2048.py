from typing import Optional
import numpy as np
import gymnasium as gymn


class Game2048Env(gymn.Env):
    metadata = {
        'render_modes': [
            'ansi'
        ]
    }

    def __init__(self, render_mode=None, size:int=4, max_score:int=2048, rate_2:float=0.9):
        self.size = size

        self._board = np.zeros([size, size], dtype=np.int32)
        self._score = 0
        self._max_score = max_score
        self.__end = False
        self.__rate_2 = rate_2

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.screen = None

        self.observation_space = gymn.spaces.Dict(
            {
                "board": gymn.spaces.Box(low=0, high=max_score, shape=(size, size), dtype=np.int32),
            }
        )

        self.action_space = gymn.spaces.Discrete(4)

        assert not self.end

    def _get_obs(self):
        return {'board': self._board}

    def _get_info(self):
        return {}

    def reset(self, seed:Optional[int]=None, options:Optional[dict]=None):
        super().reset(seed=seed)

        self._board = np.zeros((4, 4), dtype=np.int32)
        self._score = 0
        self.__end = False

        self._add_rand_tile()
        self._add_rand_tile()

        if self.render_mode == 'ansi':
            self._render_ansi()

        return self._get_obs(), self._get_info()

    @property
    def end(self):
        '''
        0: continue
        1: lose
        2: win
        '''
        if self._score >= self._max_score:
            return 2
        elif self.__end:
            return 1
        else:
            return 0

    def _add_rand_tile(self):
        where_empty = list(zip(*np.where(self._board == 0)))
        if where_empty:
            selected = self.np_random.choice(where_empty)
            self._board[tuple(selected)] = self.np_random.choice([2, 4], p=[self.__rate_2, 1 - self.__rate_2])
            self.__end = False
        else:
            self.__end = True

    def step(self, action):
        '''
        0 left
        1 down
        2 right
        3 up
        '''
        previous_board = self._board.copy()

        merge_score = 0
        # treat all direction as left (by rotation)
        board_to_left = np.rot90(self._board, -action)
        for row in range(self.size):
            # merge
            non_zero = [x for x in board_to_left[row] if x != 0]
            core = []
            i = 0
            while i < len(non_zero):
                if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                    merged_value = non_zero[i] * 2
                    core.append(merged_value)
                    merge_score += merged_value
                    i += 2
                else:
                    core.append(non_zero[i])
                    i += 1
            core += [0] * (len(board_to_left[row]) - len(core))

            board_to_left[row, :len(core)] = core
            board_to_left[row, len(core):] = 0
        # rotation to the original
        self._board = np.rot90(board_to_left, action)
        self._add_rand_tile()

        # add tile if changed
        changed = not np.array_equal(previous_board, self._board)
        if changed:
            self._add_rand_tile()

        reward = merge_score
        self._score += reward
        terminated = False
        if self.end == 1:
            # lose
            terminated = True
        elif self.end == 2:
            # win
            terminated = True
        truncated = False
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == 'ansi':
            self._render_ansi()

        return observation, reward, terminated, truncated, info

    def _render_ansi(self):
        s = ''
        s += f'score: {self._score}\n'
        for row in self._board:
            s += ('\t' + '{:8d}' * self.size + '\n').format(*map(int, row))
        print(s)

    def close(self):
        pass

