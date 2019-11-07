#!/usr/bin/python

# Solution from here:
# https://stackoverflow.com/questions/28074461/animating-growing-line-plot-in-python-matplotlib
# https://stackoverflow.com/questions/42621036/how-to-use-funcanimation-to-update-and-animate-multiple-figures-with-matplotlib

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

# Globals
stride = 20
marker = 0

        
def load_graph(log_path):
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

def animate_line(num, x, y, line):
    line.set_data(x[:num], y[:num])
    return line,


def animate_points(num, x, y, points):
    global marker
    if num >= x[marker] // stride:
        points.set_data(x[:marker+1], y[:marker+1])
        points.set_color('green')
        marker = (marker + 1) % 3
    return points,

def update_all(num, l_x, l_y, p_x, p_y, line, points):
    a = animate_line(num, l_x, l_y, line)
    b = animate_points(num, p_x, p_y, points)
    return a+b


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
    ax.set_ylim([60, 3500])

    times = times_1
    scores = scores_1
    line, = ax.plot(times, scores, color='r', lw=3)

    p_times = [50, 150, 550]
    p_scores = [600, 1000, 2000]
    points, = plt.plot(p_times, p_scores, 'o', color='white')
        
    ax.set(xlabel='time (minutes)', ylabel='scores',
       title='Scores over time')
    ax.grid()

    anim = animation.FuncAnimation(fig, update_all, frames=len(times), fargs=[times, scores, p_times, p_scores, line, points],
                                   interval=5, blit=False)
    plt.show()

        
