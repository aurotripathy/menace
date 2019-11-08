#!/usr/bin/python

# Solution from here:
# https://stackoverflow.com/questions/28074461/animating-growing-line-plot-in-python-matplotlib

import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import pandas as pd
import dateparser
from pudb import set_trace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

import matplotlib.animation as animation
from queue import Empty
from multiprocessing import Queue, Process
from multiprocessing.connection import Listener

SCORE_COL = 4
TIME_COL = 0
ROWS_TO_SKIP = 19


def receiver(q):
    print('Enter receiver')
    while 1:
        try:
            message = q.get_nowait()
            print('Receiver got:', message)
            if message == 'good':
                times = times_1
                scores = scores_1
            elif message == 'better':
                times = times_2
                scores = scores_2
                
        except Empty:
            pass
            # print('Nothing received yet....')
            # time.sleep(5)


def sender(q):

    print('start sender')
    address = ('localhost', 6000)     # family is deduced to be 'AF_INET'
    listener = Listener(address, authkey=str.encode('sc19-visuals'))
    conn = listener.accept()
    print('connection accepted from', listener.last_accepted)
    while True:
        msg = conn.recv()
        if msg in ['good', 'better', 'best']:
            print("Got message from another app:", msg)
            q.put(msg)
            print('Sender sent:', msg)
        else:
            print("Got UNKNOWN message another app:", msg)
            
        time.sleep(1)

        
def load_graph(log_path, stride=20):
    time_axis = []
    scores_axis = []
    df = pd.read_csv(log_path, header=None,
                     skiprows=ROWS_TO_SKIP)
    print('Read {} scores.'.format(len(df)))
    # set_trace()
    
    scores_axis.append(float(df[SCORE_COL][0].split()[2]))
    start_time = dateparser.parse(df[TIME_COL][0])
    time_axis.append(0)
    
    for time in range(1, len(df[0]), stride):
        time_axis.append((dateparser.parse(df[TIME_COL][time]) -
                           start_time).total_seconds() // 60)
        scores_axis.append(float(df[SCORE_COL][time].split()[2]))

    return time_axis[0:], scores_axis[0:]

def animate(num, x, y, line):
    line.set_data(x[:num], y[:num])
    return line,

def animate_points(num):
    points.set_data(points[:num], points[:num])
    return points,

def updateAll()
pass

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path" , type=str, required=True,
                        help="Full path to the log-dir")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    times_1, scores_1 = load_graph(args.log_path)
    print('len=', len(times_1))
    times_2, scores_2 = load_graph('/dockerx/data/rl/logs-150m/MsPacman-v0_log')
    print('len=', len(times_2))

    # First set up the figure, the axis, and the plot element we want to animate
    fig, ax = plt.subplots()
    ax.set_xlim([0,600])
    ax.set_ylim([60, 3000])

    times = times_1
    scores = scores_1
    line, = ax.plot(times, scores, color='r', lw=3)
    points = plt.scatter([50, 150, 550], [600, 1000, 2000], color='blue', s=20)
        
    ax.set(xlabel='time (minutes)', ylabel='scores',
       title='Scores over time')
    ax.grid()

    # Rx/Tx infra
    some_queue = Queue()
    process_sender = Process(
        target=sender,
        args=(some_queue,)
    )
    
    process_receiver = Process(
        target=receiver,
        args=(some_queue,)
    )
    
    process_sender.start()
    # process_receiver.start()

    anim = animation.FuncAnimation(fig, animate+animate_points, len(times), fargs=[times, scores, line],
                                   interval=50, blit=False)
    plt.show()
    while True:
        print('in receiver loop')
        print('short receiver:', some_queue.get())
    # receiver(some_queue)
        
