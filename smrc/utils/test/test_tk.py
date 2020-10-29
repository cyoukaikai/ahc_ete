import tkinter as tk

# if you are still working under a Python 2 version,
# comment out the previous line and uncomment the following line
# import Tkinter as tk

# root = tk.Tk()
#
# w = tk.Label(root, text="Hello Tkinter!")
# w.pack()
#
# root.mainloop()

root = tk.Tk()
logo = tk.PhotoImage(file="logo64.gif")

w1 = tk.Label(root, image=logo).pack(side="right")

explanation = """At present, only GIF and PPM/PGM
formats are supported, but an interface 
exists to allow additional image file
formats to be added easily."""

w2 = tk.Label(root,
              justify=tk.LEFT,
              padx = 10,
              text=explanation).pack(side="left")
root.mainloop()