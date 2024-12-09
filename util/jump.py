from astropy.io import fits

import tkinter as tk
from tkinter import filedialog

def open_file_dialog():
  root = tk.Tk()
  root.withdraw()
  file_path = filedialog.askopenfilename()
  return file_path

selected_file = open_file_dialog()
print(selected_file)

f1 = fits.getdata(selected_file)

selected_file = open_file_dialog()
print(selected_file)

f2 = fits.getdata(selected_file)


from util import *

print(find_optimal_scaling(f1, f2))