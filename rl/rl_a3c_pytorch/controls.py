import tkinter as tk
from tkinter import font

from three_models import process_in_sequence

root = tk.Tk()
root.title('CONTROLS')

helv36 = font.Font(family="Helvetica", size=20, weight="bold")

v = tk.IntVar()
v.set(1)  # initializing the choice, i.e. Python

languages = [
    ("Play w/Model 1"),
    ("Play w/Model 2"),
    ("Play w/Model 3"),
    ("AutoPlay All"),
]

def process_choice():
    print('You selected:', v.get())
    process_in_sequence()
    
tk.Label(root, font=helv36,
         text="""Choose Model:""",
         justify = tk.LEFT,
         padx = 20).pack()

for val, language in enumerate(languages):
    tk.Radiobutton(root, 
            text=language,
            font=helv36,
            indicatoron = 0,  # indicator on
            width = 20,
            padx = 20, 
            variable=v, 
            command=process_choice,
            value=val).pack(anchor=tk.W)


root.mainloop()
