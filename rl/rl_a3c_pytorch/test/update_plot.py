import matplotlib.pyplot as plt
import numpy as np
import time
from multiprocessing.connection import Listener
from pudb import set_trace

X_LIM = 100
Y_LIM = 60

def build_graph():
    import numpy as np
    l1_x = np.arange(0, 30)
    l1_y = l1_x
    l2_x = np.arange(30, 60)
    l2_y = (l2_x - 30) * 0.5 + l1_y[-1]
    l3_x = np.arange(60, 90)
    l3_y = (l3_x - 60) * 0.25 + l2_y[-1]
    
    return np.concatenate((l1_x, l2_x, l3_x)), np.concatenate((l1_y, l2_y, l3_y))



address = ('localhost', 6000)     # family is deduced to be 'AF_INET'
listener = Listener(address, authkey=str.encode('sc19-visuals'))


# You probably won't need this if you're embedding things in a tkinter plot...
plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim([0, X_LIM])
ax.set_ylim([0, Y_LIM])
plt.pause(0.001)

x, y = build_graph()
line1, = ax.plot(x[:0], y[:0], 'r-') # Returns a tuple of line objects, thus the comma
fig.canvas.draw()
fig.canvas.flush_events()

seq = [30, 60, 90, 0]
conn = listener.accept()
i = 0
while True:
    print('waiting for messages')
    i = i % 4
    if seq[i] == 0: #  starting point
        ax.clear()
    msg = conn.recv()
    print('got message ', msg)
    if msg == 'next':
        print('updating...', seq[i])
        ax.set_xlim([0, X_LIM])
        ax.set_ylim([0, Y_LIM])
        line1, = ax.plot(x[:seq[i]], y[:seq[i]], 'r-') # Returns a tuple of line objects, thus the comma
        plt.pause(0.001)
        i += 1


