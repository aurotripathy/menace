import tkinter as tk
from tkinter import font

root = tk.Tk()

helv36 = font.Font(family="Helvetica", size=36, weight="bold")

v = tk.IntVar()
v.set(1)  # initializing the choice, i.e. Python

languages = [
    ("Play in Sequence"),
    ("Play Model 1"),
    ("Play Model 2"),
    ("Play Model 3"),
    ("Play Model 4")
]

def ShowChoice():
    print('You selecled:', v.get())

tk.Label(root, font=helv36,
         text="""Choose your Model:""",
         justify = tk.LEFT,
         padx = 20).pack()

for val, language in enumerate(languages):
    tk.Radiobutton(root, 
            text=language,
            font=helv36,
            indicatoron = 0,
            width = 20,
            padx = 20, 
            variable=v, 
            command=ShowChoice,
            value=val).pack(anchor=tk.W)


root.mainloop()
