#!/usr/bin/python
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import pandas as pd
import dateparser
from pudb import set_trace

SCORE_COL = 4
TIME_COL = 0
def refresh_graph():
    time_axis = []
    scores_axis = []
    df = pd.read_csv('./../logs/MsPacman-v0_log', header=None, skiprows=19)
    print('Read {} score.'.format(len(df)))
    # set_trace()
    
    scores_axis.append(float(df[SCORE_COL][0].split()[2]))
    start_time = dateparser.parse(df[TIME_COL][0])
    time_axis.append(0)
    
    for time in range(1, len(df[0])):
        time_axis.append((dateparser.parse(df[TIME_COL][time]) -
                           start_time).total_seconds())
        scores_axis.append(float(df[SCORE_COL][time].split()[2]))

    print('{}'.format([[t, s] for t, s in zip(time_axis[-10:], scores_axis[-10:])]))

class MyHandler(FileSystemEventHandler):
    def on_modified(self, event):
        print(f'event type: {event.event_type}  path : {event.src_path}')
        refresh_graph()


if __name__ == "__main__":
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
