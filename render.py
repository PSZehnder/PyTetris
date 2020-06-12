import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from mpl_toolkits.mplot3d import art3d
from .shapes import Tetromino
import cv2
import matplotlib.animation as ani

# render a queue of frames
def render2d(frames, fig, ax, path=None, boxsize=25, interval=10,):
    assert frames, 'must have at least one frame'
    size = frames[0].shape
    target_size = tuple([dim * boxsize for dim in size])

    def draw_frame(frame):
        im = cv2.resize(np.flip(frame, axis=1), target_size)
        ax.imshow(im, animated=True)

    anim = ani.FuncAnimation(fig, draw_frame, frames=frames, interval=interval, repeat=False)

    if path:
        anim.save(path)
    else:
        plt.show()

def render3d(frames, fig, ax, path=None, interval=10):

    def draw_frame(frame):
        [p.remove() for p in reversed(ax.collections)]
        frame = np.flip(frame, axis=2)
        cols = generate_external_faces(frame)
        for i in cols:
            ax.add_collection3d(i)

    anim = ani.FuncAnimation(fig, draw_frame, frames=frames, interval=interval, repeat=False)

    if path:
        anim.save(path)
    else:
        plt.show()

def init3dplot(dimensions):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    ax.set_xlim(0, dimensions[0])
    ax.set_ylim(0, dimensions[1])
    ax.set_zlim(0, dimensions[2])
    return fig, ax

# returns a list of Poly3D collections, so need to use a forloop to unlist
def generate_external_faces(board_array):

    shape = board_array.shape
    collection = []

    def generate_face(location, color1, axis1, direction1):
        x_face = [[0, 0, 0],
                  [0, 0, 1],
                  [0, 1, 1],
                  [0, 1, 0]]

        y_face = [[0, 0, 0],
                  [1, 0, 0],
                  [1, 0, 1],
                  [0, 0, 1]]

        z_face = [[0, 0, 0],
                  [1, 0, 0],
                  [1, 1, 0],
                  [0, 1, 0]]

        faces = [x_face, y_face, z_face]

        face = faces[axis1]
        for row in face:
            if direction1 == 1:
                row[axis1] += 1
            for i in range(3):
                row[i] += location[i]

        face_nump = np.array(face)
        side = art3d.Poly3DCollection([face_nump])
        side.set_edgecolor('k')
        side.set_facecolor(color1)
        collection.append(side)

    # iterate through the array and check to see if neighbors. If no neighbor, generate face
    for x, y, z in product(*[range(shape[i]) for i in range(3)]):
        if board_array[x, y, z] != 0:
            color = Tetromino.COLOR_VEC[int(board_array[x, y, z])]

            for axis, direction in product(range(3), [-1,1]):
                temp = [x, y, z]
                if 0 <= temp[axis] + direction < shape[axis]:
                    temp[axis] += direction
                    if board_array[tuple(temp)] == 0:
                        generate_face([x, y, z], color, axis, direction)

                else:  # its an edge face, so draw it
                    generate_face([x, y, z], color, axis, direction)

    return collection