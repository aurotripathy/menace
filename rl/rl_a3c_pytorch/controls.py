import tkinter as tk
from tkinter import font
import subprocess

from three_models import process_in_sequence

cmd_play_all = "python3.6 three_models.py".split()

root = tk.Tk()
root.title('CONTROLS')

INITIAL_SELECTION = 3
current_selection = None


# helv20 = font.Font(family="Helvetica", size=20, weight="bold")
helv20 = font.Font(family="Helvetica", size=20)

v = tk.IntVar()
v.set(INITIAL_SELECTION)  

options = [
    ("50 min Checkpoint"),
    ("150 min Checkpoint"),
    ("550 min Checkpoint"),
    ("AutoPlay Checkpoints"),
]

def process_choice():
    global current_selection
    print('current selection {}, You selected: {}'.format(current_selection, v.get()))
    if v.get() != current_selection:
        if v.get() == 3:
            sub_p = subprocess.Popen(cmd_play_all)
            print("process pid:", sub_p.pid)
            # process_in_sequence()
    else:
        print("No processing required!")
    current_selection = v.get()

    
tk.Label(root, font=helv20,
         text="""Choose Model:""",
         justify = tk.LEFT,
         padx = 20).pack()

for val, option in enumerate(options):
    tk.Radiobutton(root, 
            text=option,
            font=helv20,
            indicatoron = 0,  # indicator on, full txt in box
            width = 20,
            padx = 20, 
            variable=v, 
            command=process_choice,
            value=val).pack(anchor=tk.W)


root.mainloop()
