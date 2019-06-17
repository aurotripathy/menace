import os

class CheckPointHandler():
    def __init__(self):
        super().__init__()
        
    def on_modified(self, event):
        self.process(event)

    def dispatch(self, event):
        filename, ext = os.path.splitext(event.src_path)
        print(filename, ext)

