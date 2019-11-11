import tkinter as tk
from tkinter import font

from three_models import process_in_sequence

root = tk.Tk()
root.title('CONTROLS')

# helv20 = font.Font(family="Helvetica", size=20, weight="bold")
helv20 = font.Font(family="Helvetica", size=20)

v = tk.IntVar()
v.set(3)  # initializing the choice, i.e. Python

options = [
    ("Play w/Model 1"),
    ("Play w/Model 2"),
    ("Play w/Model 3"),
    ("AutoPlay All Models"),
]

def process_choice():
    print('You selected:', v.get())
    if v.get() == 3:
        process_in_sequence()
    
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
