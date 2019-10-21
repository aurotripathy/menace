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

SCORE_COL = 4
TIME_COL = 0
ROWS_TO_SKIP = 19

def display_two(time_1, scores_1,
                time_2, scores_2,
                save_path):

    fig, ax = plt.subplots()
    ax.plot(time_1, scores_1,
            time_2, scores_2)

    ax.set(xlabel='time (s)', ylabel='scores',
       title='Scores over time')
    ax.grid()

    fig.savefig(save_path)
    plt.show()

def get_time_scores(df):
    time_axis = []
    scores_axis = []
    scores_axis.append(float(df[SCORE_COL][0].split()[2]))
    start_time = dateparser.parse(df[TIME_COL][0])
    time_axis.append(0)
    
    for time in range(1, len(df[0])):
        time_axis.append((dateparser.parse(df[TIME_COL][time]) -
                           start_time).total_seconds())
        scores_axis.append(float(df[SCORE_COL][time].split()[2]))

    print('{}'.format([[t, s] for t, s in zip(time_axis[-10:], scores_axis[-10:])]))
    return time_axis, scores_axis


    
def refresh_graphs(log_path_1, log_path_2):
    df_1 = pd.read_csv(log_path_1, header=None,
                     skiprows=ROWS_TO_SKIP)
    df_2 = pd.read_csv(log_path_2, header=None,
                     skiprows=ROWS_TO_SKIP)
    print('Read {} scores 1.'.format(len(df_1)))
    print('Read {} scores 2.'.format(len(df_2)))
    # set_trace()
    
    time_axis_1, scores_axis_1 = get_time_scores(df_1)
    time_axis_2, scores_axis_2 = get_time_scores(df_2)
    
    save_path = os.path.join(os.path.dirname(log_path_1), 'scores_plot.png')
    display_two(time_axis_1[-300:], scores_axis_1[-300:],
                time_axis_2[-300:], scores_axis_2[-300:],
                save_path)

class MyHandler(FileSystemEventHandler):
    def on_modified(self, event):
        print(f'event type: {event.event_type}  path : {event.src_path}')
        refresh_graph()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path-1" , type=str, required=True,
                        help="Full path to the log-path 1")
    parser.add_argument("--log-path-2" , type=str, required=True,
                        help="Full path to the log-path 2")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    refresh_graphs(args.log_path_1, args.log_path_2)
    event_handler = MyHandler()
    observer = Observer()
    observer.schedule(event_handler, path='./../logs', recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
