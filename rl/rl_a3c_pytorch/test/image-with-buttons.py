import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# to easily change example
DEFAULT = "./../logos/PyTorch_sw_stack_3.png"
PHOTO_1 = "./../logos/EPYC-blue-white.png"
PHOTO_2 = "./../logos/EPYC-blue-white.png"


def change_image():
    print(photo_filepath.get())

    photo = tk.PhotoImage(file=photo_filepath.get())
    calc_button['image'] = photo
    calc_button.image = photo # solution for garbage-collector problem. you have to assign PhotoImage object to global variable or class variable

    # - or -

    photo['file'] = photo_filepath.get()
    calc_button['image'] = photo


root = tk.Tk()  # Set up 
root.title("Test GUI")

photo_filepath = tk.StringVar()  # Set default photo
photo_filepath.set(DEFAULT)

# photo = tk.PhotoImage(file=photo_filepath.get())
photo = Image.open('./../logos/PyTorch_sw_stack_3.png')
photo = ImageTk.PhotoImage(photo)

calc_button = ttk.Button(root, image=photo)
calc_button.grid(column=3, row=2, columnspan=1)

photo1_radiobutton = ttk.Radiobutton(root, text="Photo 1", variable=photo_filepath,
                                  value=PHOTO_1, command=change_image)
photo1_radiobutton.grid(column=4, row=2, sticky=tk.S)

photo2_radiobutton = ttk.Radiobutton(root, text="Photo 2", variable=photo_filepath,
                                  value=PHOTO_2, command=change_image)
photo2_radiobutton.grid(column=4, row=3)

root.mainloop()
