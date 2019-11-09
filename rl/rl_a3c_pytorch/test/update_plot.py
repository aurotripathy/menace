import matplotlib.pyplot as plt
import numpy as np
import time
from pudb import set_trace

def build_graph():
    import numpy as np
    set_trace()
    l1_x = np.arange(0, 30)
    l1_y = l1_x
    l2_x = np.arange(30, 60)
    l2_y = (l2_x - 30) * 0.5 + l1_y[-1]
    l3_x = np.arange(60, 90)
    l3_y = (l3_x - 60) * 0.25 + l2_y[-1]
    
    return np.concatenate((l1_x, l2_x, l3_x)), np.concatenate((l1_y, l2_y, l3_y))



# You probably won't need this if you're embedding things in a tkinter plot...
plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim([0,100])
ax.set_ylim([0, 80])

x, y = build_graph()
line1, = ax.plot(x[:0], y[:0], 'r-') # Returns a tuple of line objects, thus the comma


fig.canvas.draw()
fig.canvas.flush_events()
# time.sleep(2)
while True:
    line1, = ax.plot(x[:30], y[:30], 'r-') # Returns a tuple of line objects, thus the comma
    inp = input('something')
    line1, = ax.plot(x[:60], y[:60], 'r-') # Returns a tuple of line objects, thus the comma
    inp = input('something')
    line1, = ax.plot(x[0:], y[0:], 'r-') # Returns a tuple of line objects, thus the comma
    inp = input('something')
    ax.clear()
    ax.set_xlim([0,100])
    ax.set_ylim([0, 80])
    # line1, = ax.plot(x[:0], y[:0], 'r-') # Returns a tuple of line objects, thus the comma
