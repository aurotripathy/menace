#!/usr/bin/python

import time

import pandas as pd
import dateparser
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

from multiprocessing import Queue, Process
from multiprocessing.connection import Listener
from pudb import set_trace

SCORE_COL = 4
TIME_COL = 0
ROWS_TO_SKIP = 19

# Globals
stride = 1
marker = 0
X_LIM = 750
Y_LIM = 3500
        
def load_graph(log_path):
    time_axis = []
    scores_axis = []
    df = pd.read_csv(log_path, header=None,
                     skiprows=ROWS_TO_SKIP)
    print('Read {} scores.'.format(len(df)))

    
    scores_axis.append(float(df[SCORE_COL][0].split()[2]))
    start_time = dateparser.parse(df[TIME_COL][0])
    time_axis.append(0)
    
    for time in range(1, len(df[0]), stride):
        time_axis.append((dateparser.parse(df[TIME_COL][time]) -
                           start_time).total_seconds() // 60)
        scores_axis.append(float(df[SCORE_COL][time].split()[2]))

    return time_axis[0:], scores_axis[0:]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path" , type=str, required=False,
                        help="Full path to the log-dir")
    return parser.parse_args()

if __name__ == "__main__":

    address = ('localhost', 6000)     # family is deduced to be 'AF_INET'
    listener = Listener(address, authkey=str.encode('sc19-visuals'))

    args = get_args()
    times_list = []; scores_list = []
    
    times_1, scores_1 = load_graph('/dockerx/data/rl/logs-53m/MsPacman-v0_log')
    times_list.append(times_1); scores_list.append(scores_1)
    print('len=', len(times_1))
    
    times_2, scores_2 = load_graph('/dockerx/data/rl/logs-150m/MsPacman-v0_log')
    print('len=', len(times_2))
    times_list.append(times_2); scores_list.append(scores_2)

    times_3, scores_3 = load_graph('/dockerx/data/rl/logs-550m/MsPacman-v0_log')
    print('len=', len(times_3))
    times_list.append(times_3); scores_list.append(scores_3)
    
    # times_4, scores_4 = load_graph('/dockerx/data/rl/logs-690/MsPacman-v0_log')
    # print('len=', len(times_4))
    # times_list.append(times_4); scores_list.append(scores_4)
    
    
    fig = plt.figure(facecolor='black')
    plt.rcParams['toolbar'] = 'None' # Remove tool bar (upper bar)
    fig.canvas.set_window_title('RL TRAINING')
    ax = fig.add_subplot(111)
    ax.set_facecolor("black")
    ax.set_xlim(0, X_LIM)
    ax.set_ylim(0, Y_LIM)  # change
    plt.pause(0.001)

    # set_trace()

    conn = listener.accept()
    i = 0
    while True:
        print('waiting for messages')
        i = i % 3
        # if i == 0: #  starting point
        #     ax.clear()
        ax.clear()
        msg = conn.recv()
        print('got message ', msg)
        if msg == 'next':
            print('updating...', i)
            ax.set_xlim(0, X_LIM)
            ax.set_ylim(0, Y_LIM)  # change
            line1, = ax.plot(times_list[i], scores_list[i], 'r-') # Returns a tuple, thus the comma
            plt.pause(0.001)
            i += 1


