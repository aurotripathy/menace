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

def display(time, scores, save_path):

    fig, ax = plt.subplots()
    ax.plot(time, scores)

    ax.set(xlabel='time (s)', ylabel='scores',
       title='Scores over time')
    ax.grid()

    fig.savefig(save_path)
    plt.show()

def refresh_graph(log_path):
    time_axis = []
    scores_axis = []
    df = pd.read_csv(log_path, header=None,
                     skiprows=ROWS_TO_SKIP)
    print('Read {} scores.'.format(len(df)))
    # set_trace()
    
    scores_axis.append(float(df[SCORE_COL][0].split()[2]))
    start_time = dateparser.parse(df[TIME_COL][0])
    time_axis.append(0)
    
    for time in range(1, len(df[0])):
        time_axis.append((dateparser.parse(df[TIME_COL][time]) -
                           start_time).total_seconds())
        scores_axis.append(float(df[SCORE_COL][time].split()[2]))

    print('{}'.format([[t, s] for t, s in zip(time_axis[-10:], scores_axis[-10:])]))
    save_path = os.path.join(os.path.dirname(log_path), 'scores_plot.png')
    display(time_axis[0:], scores_axis[0:], save_path)

class MyHandler(FileSystemEventHandler):
    def on_modified(self, event):
        print(f'event type: {event.event_type}  path : {event.src_path}')
        refresh_graph()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path" , type=str, required=True,
                        help="Full path to the log-dir")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    refresh_graph(args.log_path)
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
