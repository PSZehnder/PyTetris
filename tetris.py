import time
from random import sample
from .render import *
from .shapes import *


# June 2020 refactor/clean up

class TetrisInstance:

    def __init__(self, board_shape):
        self.board_extents = [x for x in board_shape if x > 1]
        self.dimension = len(self.board_extents)

        self.pieces = Shapes2d if self.dimension < 3 else Shapes3d
        self.current = self.get_new_piece()
        self.next = self.get_new_piece()

        self.board = np.zeros(tuple(self.board_extents))
        self.board_with_piece = np.copy(self.board)
        self.done = False

        self.board_size = 1
        for dim in self.board_extents:
            self.board_size = self.board_size * dim

        self.score = 0
        self.cumscore = 0
        self.total_pieces = 0

        self.start_time = time.time()

        # after some ticks, we force a drop
        self.drop_tick_counter = 0
        self.ticks = 0

        self.game_over_penalty = -1
        self.passive_reward = 1
        self.clear_reward = 5

        # compute action space
        self.action_space = actionspace(self.dimension)

        # action space cardinality
        self.output_dimension = len(self.action_space)

    def compute_reward(self, n=0):

        clear_reward = self.clear_reward * n ** 2
        game_over_penalty = int(self.done) * (-self.game_over_penalty)
        passive_reward = int(not self.done) * self.passive_reward

        return clear_reward + game_over_penalty + passive_reward

    def get_new_piece(self):
        shape = sample(self.pieces)
        location = [x // 2 for x in self.board_extents[:-1]] + [0]
        return Tetromino(shape, location)

    # adds the current piece to the board. If inplace, then directly modifies self.board, then returns self.board;
    # Otherwise, makes a copy, then returns the modified copy. by defualt, set to 1
    def add_to_board(self, inplace=True, color=None):
        if not inplace:
            temp = copy.deepcopy(self.board)
        else:
            temp = self.board
        if not color:
            color = self.current.color

        coords = product(*[range(self.current.shape[i]) for i in range(self.dimension)])

        for tup in coords:
            if self.current.matrix[tup] != 0:
                _idx = tuple([j + self.current.location[i] for (i, j) in enumerate(tup)])
                temp[_idx] = color

        return temp

    # simple bounds check for a given location (tuple)
    def is_on_board(self, loc):
        temp = all([0 <= loc[i] < self.board_extents[i] for i in range(self.dimension - 1)])
        return temp and loc[-1] < self.board_extents[-1]

    # checks if the piece with a given offset is within the bound and not colliding
    def is_valid_position(self, newmat, offset=None):

        coords = product(*[range(newmat.shape[i]) for i in range(self.dimension)])

        for tup in coords:
            if newmat[tup] == 0:
                continue

            _idx = tuple([j + offset[i] for (i, j) in enumerate(tup)])

            if not self.is_on_board(_idx) or self.board[_idx] != 0:
                return False
        return True

    def is_complete_line(self, z): return np.any(self.board[..., z] == 0)

    # remove the complete lines and drop the ones above it in the classic tetris fashion; return num consec lines
    # cleared all at once
    def remove_complete_lines(self):
        consec_clears = 0
        for z in range(self.board_extents[-1]):
            if self.is_complete_line(z):
                np.delete(z, self.board_extents[-1] - 1) # delete row
                consec_clears += 1

        self.board = np.reshape(self.board, self.board_extents)
        return consec_clears

    def is_valid_action(self, action):

        def check_translate(axis, direction):
            offset = [0, 0, 0]
            offset[axis] = direction
            temp = copy.copy(self.current.location)
            temp[axis] += direction

            return self.is_valid_position(self.current.matrix, offset)

        def check_rotate(axis, direction=None):
            axes = [0, 1, 2]
            del axes[axis]
            temp = np.rot90(self.current.matrix, k=direction, axes=axes)
            return self.is_valid_position(temp)

        def check_drop(axis=None, direction=None):
            return True

        switcher = {
            'translate': check_translate,
            'rotate': check_rotate,
            'drop': check_drop
        }

        return switcher[action.type](action.axis, action.direction)

    # must pass Action objects; transitions the board
    def update(self, action):

        self.score = 0
        self.ticks += 1

        # action block: each takes an axis and direction
        def translate(axis, direction):
            offset = [0, 0, 0]
            offset[axis] = direction
            temp = copy.copy(self.current.location)
            temp[axis] += direction

            if self.is_valid_position(self.current.matrix, offset):
                self.current.location = temp

        def rotate(axis, direction):
            if axis == None or direction == None:
                assert self.dimension == 2
                temp = np.rot90(self.current.matrix)
            else:
                axes = [0, 1, 2]
                del axes[axis]
                temp = np.rot90(self.current.matrix, k=direction, axes=axes)
            if self.is_valid_position(temp):
                self.current.matrix = temp
                self.current.shape = temp.shape

        offset = (0, 1) if self.dimension == 2 else (0, 0, 1)

        def drop(axis=None, direction=None):
            if self.is_valid_position(self.current.matrix, offset):
                self.current.location[-1] += 1

        switcher = {
            'translate': translate,
            'rotate': rotate,
            'drop': drop
        }

        switcher[action.type](action.axis, action.direction)

        self.drop_tick_counter += 1

        if self.drop_tick_counter >= 3:
            drop()
            self.drop_tick_counter = 0

        self.board_with_piece = self.add_to_board(inplace=False, color=100)

        self.reward = self.compute_reward()

        # check the board
        if not self.is_valid_position(self.current.matrix, offset):
            self.board = self.add_to_board(color=1)
            removed_lines = self.remove_complete_lines()
            self.reward = self.compute_reward(n=removed_lines)
            self.cumscore += self.reward
            self.current = self.next
            self.next = self.get_new_piece()
            self.total_pieces += 1
            if not self.is_valid_position(self.current.matrix):
                self.done = True
                self.reward = self.compute_reward(removed_lines)

    # returns a clean version of the game state with the same parameters
    def reset(self):
        self.current = self.get_new_piece()
        self.next = self.get_new_piece()
        self.board = np.zeros(tuple(self.board_extents))
        self.done = False

        self.score = 0
        self.cumscore = 0
        self.total_pieces = 0

        self.drop_tick_counter = 0
        self.ticks = 0

        self.start_time = time.time()
        return self

    # This is the simulator function Q-Learner will call (implementing this as a __call__ method
    # cleans up the syntax quite a bit)
    def __call__(self, action, verbose=False):
        if verbose:
            print(action)
        self.update(action)
        return self, self.reward, self.done


class Action:

    def __init__(self, typ, axis=None, direction=None):
        if axis:
            if axis > 2:
                raise Exception('Action: axis %d out of bounds' % axis)
        if direction:
            if direction != 1 and direction != -1:
                raise Exception('Action: direction %d out of bounds' % direction)

        self.axis = axis
        self.direction = direction
        self.type = typ

    def __repr__(self):
        return '%s (%s %s)' % (self.type, self.axis, self.direction)

def actionspace(dimension):
    if dimension == 2:
        return [Action('drop'), Action('rotate'), Action('translate', direction=1),
                Action('translate', direction=-1)]
    if dimension == 3:
        out = [Action('drop')] + [Action('translate', i, j) for i, j in product(range(2), (-1, 1))]
        out = out + [Action('rotate', i, j) for i, j in product(range(3), (-1, 1))]
        return out
    else:
        raise ValueError('dimension %d out of bounds' % dimension)
