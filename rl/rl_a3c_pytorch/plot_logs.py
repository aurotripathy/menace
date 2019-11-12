#!/usr/bin/python
# branding from https://ramiro.org/notebook/matplotlib-branding/
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
import matplotlib.image as image
from pudb import set_trace

SCORE_COL = 4
TIME_COL = 0
ROWS_TO_SKIP = 19

# Globals
stride = 1
marker = 0
X_LIM = 600
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

def refresh_window_dressing(ax, plt):
    ax.set_xlim(0, X_LIM)
    ax.set_ylim(0, Y_LIM)  # change
    plt.xlabel('Training Time (minutes)\n100 cores + 8 GPUs', fontsize=16, color='white')
    plt.ylabel('Scores', fontsize=16, color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.grid(linestyle='-', linewidth='0.5', color='white')
    ax.figure.figimage(logo, 200, 100, alpha=.80, zorder=1)
    

if __name__ == "__main__":
    logo = image.imread('/dockerx/data/rl/logos/AMD.png')
    address = ('localhost', 6000)     # family is deduced to be 'AF_INET'
    listener = Listener(address, authkey=str.encode('sc19-visuals'))

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
    
    plt.rcParams['toolbar'] = 'None' # needs to done before instantiation
    plt.rc('axes',edgecolor='white')
    fig = plt.figure(figsize=(6, 6.30), facecolor='black')
    fig.suptitle('Scores Over Time', fontsize=18, color='white')
    fig.canvas.set_window_title('RL TRAINING')
    ax = fig.add_subplot(111)
    ax.set_facecolor("black")
    refresh_window_dressing(ax, plt)
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
            refresh_window_dressing(ax, plt)
            line, = ax.plot(times_list[i], scores_list[i], 'yellow', linewidth=3) # Returns a tuple, thus the comma
            plt.plot(times_list[i][-1], scores_list[i][-1], marker='o', markersize=4, color="red")
            plt.pause(0.001)
            i += 1


