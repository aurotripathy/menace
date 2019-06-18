import sys
import time
import os

from watchdog.observers import Observer
# from events import CheckPointHandler

from watchdog.events import RegexMatchingEventHandler

sleep_time = -1  # Global scope

class CheckPointHandler(RegexMatchingEventHandler):

    MODEL_REGEX = [r".*[^_thumbnail]\.dat$"]
    
    def __init__(self):
        super().__init__(self.MODEL_REGEX)
        
    def on_modified(self, event):
        print('Event:', event)
        self.process(event)

    def process(self, event):
        global sleep_time
        filename, ext = os.path.splitext(event.src_path)
        print('Process: Filename', filename, 'Extension', ext)
        with open('sleep.dat') as f:
            sleep_time = int(f.read())


def do_normal_processing():
    global sleep_time
    with open('sleep.dat') as f:
        sleep_time = f.read()
    while True:
        print('Sleeping for {} seconds'.format(sleep_time))
        time.sleep(int(sleep_time))
    

class CheckPointWatcher:
    def __init__(self, src_path):
        self.__src_path = src_path
        self.__event_handler = CheckPointHandler()
        self.__event_observer = Observer()

    def run(self):
        self.start()
        try:
            do_normal_processing()
        except KeyboardInterrupt:
            self.stop()


    def start(self):
        self.__schedule()
        self.__event_observer.start()

    def stop(self):
        self.__event_observer.stop()
        self.__event_observer.join()

    def __schedule(self):
        self.__event_observer.schedule(
            self.__event_handler,
            self.__src_path,
            recursive=True
        )

if __name__ == "__main__":
    src_path = sys.argv[1] if len(sys.argv) > 1 else '.'
    CheckPointWatcher(src_path).run()
