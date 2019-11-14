import tkinter as tk
from tkinter import font
import subprocess
import psutil
import time
from PIL import Image, ImageTk


play_template = "python3.6 launch_player.py --control-option xx".split()
plot_cmd = "python3.6 plot_logs.py".split()

root = tk.Tk()
root.title('CONTROLS')
frame = tk.Frame(root)

INITIAL_SELECTION = 3
current_selection = None
sub_p = None

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

def clean_up():
    # find the processes to terminate
    for process in psutil.process_iter():
        app_with_params = process.cmdline()
        if app_with_params[:2] == ['python3.6', 'gym-matplotlib-animated-eval.py']:
            print('Process gym-matplotlib-animated-eval.py found. Terminating it.')
            process.terminate()
            break
    for process in psutil.process_iter():
        app_with_params = process.cmdline()
        if app_with_params[:2] == ['python3.6', 'launch_player.py']:
            print('Found Process launch_player.py. Terminating it.')
            process.terminate()
            break


def process_choice():
    global current_selection
    global sub_p
    print('current selection {}, You selected: {}'.format(current_selection, v.get()))
    if v.get() != current_selection:
        clean_up()
        if v.get() == 3:
            plot_process = subprocess.Popen(plot_cmd)
            time.sleep(1)
            play_template[-1] = '3'
            sub_p = subprocess.Popen(play_template)
            print("process pid:", sub_p.pid)
        elif v.get() in [0, 1, 2]:
            play_template[-1] = str(v.get())
            sub_p = subprocess.Popen(play_template)

    else:
        print("No processing required!")
    current_selection = v.get()

IMAGE_LOC = "./logos/PyTorch_sw_stack_3.png"
photo = Image.open(IMAGE_LOC)
photo = ImageTk.PhotoImage(photo)

tk.Label(root, font=helv20,
         image=photo,
         justify = tk.CENTER,
         padx = 20).pack()
tk.Label(root, font=helv20,
         text="""Choose Model:""",
         justify = tk.CENTER,
         padx = 20).pack()

for val, option in enumerate(options):
    tk.Radiobutton(root, 
            text=option,
            font=helv20,
            indicatoron = 0,  # indicator on, full txt in box
            width = 26,
            padx = 20,
            justify = tk.CENTER,
            variable=v, 
            command=process_choice,
            value=val).pack(anchor=tk.W)


root.mainloop()
