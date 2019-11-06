#!/usr/bin/python
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


SCORE_COL = 4
TIME_COL = 0
ROWS_TO_SKIP = 19


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
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path" , type=str, required=True,
                        help="Full path to the log-dir")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    times, scores = load_graph(args.log_path)

    # First set up the figure, the axis, and the plot element we want to animate
    fig, ax = plt.subplots()
    line, = ax.plot(times, scores, color='r')

    ax.set(xlabel='time (minutes)', ylabel='scores',
       title='Scores over time')
    ax.grid()

    ani = animation.FuncAnimation(fig, animate, len(times), fargs=[times, scores, line],
                                  interval=75, blit=False)    
    
    plt.show()
