"""
===========================================
Open AI gym running as matplotlib animation
===========================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gym

fig = plt.figure()
env = gym.make('MsPacman-v0')
env.reset()
im = plt.imshow(env.render(mode='rgb_array'), animated=True)


def updatefig(frame, *fargs):
    global anim
    print(frame, fargs)
    im.set_data(env.render(mode='rgb_array'))
    action = env.action_space.sample()
    _, _, done, _ = env.step(action)
    if done:
        env.reset()
        anim.event_source.stop()
    # return im,

status ='running'
anim = animation.FuncAnimation(fig, updatefig, fargs=(1,2,3), frames=10000, interval=10, blit=False, repeat=False)
plt.show()

    
