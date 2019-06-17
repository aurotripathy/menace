import os
from watchdog.events import RegexMatchingEventHandler


class CheckPointHandler(RegexMatchingEventHandler):

    MODEL_REGEX = [r".*[^_thumbnail]\.dat$"]
    
    def __init__(self):
        super().__init__(self.MODEL_REGEX)
        
    def on_modified(self, event):
        print('Event:', event)
        self.process(event)

    def process(self, event):
        filename, ext = os.path.splitext(event.src_path)
        print('Process: Filename', filename, 'Extension', ext)

