"""
=================
An animated image
=================

This example demonstrates how to animate an image.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gym

fig = plt.figure()
env = gym.make('MsPacman-v0')
env.reset()
im = plt.imshow(env.render(mode='rgb_array'), animated=True)


def updatefig(*args):
    im.set_data(env.render(mode='rgb_array'))
    action = env.action_space.sample()
    _, _, done, _ = env.step(action)
    if done:
        env.reset()
    return im,


ani = animation.FuncAnimation(fig, updatefig, interval=10, blit=True)
plt.show()
