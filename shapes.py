import numpy as np
import enum

# Array representations of our pieces
@enum.unique
class Shapes3d(enum.Enum):

    STR = np.array([[[1, 1, 1, 1]]])
    L   = np.array([[[1, 1, 1],
                     [1, 0, 0]]])
    S   = np.array([[[1, 1, 0],
                     [0, 1, 1]]])
    SQ  = np.array([[[1, 1],
                     [1, 1]]])
    T   = np.array([[[1, 1, 1],
                     [0, 1, 0]]])
    RSK = np.array([[[1, 1], [0, 1]],
                    [[0, 0], [0, 1]]])
    LSK = np.array([[[1, 1], [1, 0]],
                    [[0, 0], [1, 0]]])
    BR  = np.array([[[1, 1], [0, 1]],
                    [[0, 1], [0, 0]]])

@enum.unique
class Shapes2d(enum.Enum):

    STR = np.array([[1, 1, 1, 1]])
    RL  = np.array([[1, 1, 1],
                    [1, 0, 0]])
    LL  = np.array([[1, 1, 1],
                    [0, 0, 1]])
    RS = np.array([[0, 1, 1],
                   [1, 1, 0]])
    LS = np.array([[1, 1, 0],
                   [0, 1, 1]])
    SQ = np.array([[1, 1],
                   [1, 1]])

# standardize the matricies by embedding them in a matrix of embedding_shape
def embed(matrix, embedding_shape=None):
    if embedding_shape is None:
        dim = matrix.ndim
        embedding_shape = tuple([4] * dim)
    embedding_matrix = np.zeros(shape=embedding_shape)
    embedding_matrix[0:matrix.shape[0], 0:matrix.shape[1], 0:matrix.shape[2]] = matrix
    return embedding_matrix

# generate a random block that fits within the bounding shape with density proportion of them filled (average)
# note that by construction: embed(generate_random_block(bounding_shape)) == generate_random_block(bounding_shape)
def _generate_random_block(bounding_shape, density=0.25):
    matrix = np.zeros(bounding_shape)
    current = [0, 0, 0]
    done = False
    while not done:
        matrix[tuple(current)] = 1
        axis = np.random.choice(range(4), p=[(1 - density)/3] * 3 + [density])
        if axis == 3 or not (matrix == 0).any():
            return matrix
        direction = np.random.choice([-1, 1])
        if current[axis] + direction in range(bounding_shape[axis]):
            current[axis] += direction

class Tetromino:

    COLOR_VEC = ("r", "g", "b", "c", "m", "k", (1, 0.5, 0.5), (0.5, 0.5, 1), (0.75, 0.25, 0.1))

    def __init__(self, matrix, location, embedding=None):
        self.location = np.array(location)
        self.matrix = embed(matrix, embedding_shape=embedding)
        self.color = np.random.randint(1, len(self.COLOR_VEC) - 1)
        self.shape = matrix.shape
        self.size = 1
        for d in embedding:
            self.size *= d
