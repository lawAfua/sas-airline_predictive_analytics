# importing tkinter and tkinter.ttk 
# and all their functions and classes 
from tkinter import *
from tkinter.ttk import *

# importing askopenfile function 
# from class filedialog 
from tkinter.filedialog import askopenfile

root = Tk()
root.title('Open SAS files for training')

root.geometry('500x200')


# This function will be used to open
# file in read mode and only Python files 
# will be opened 
def open_shipping_request_inp():
    file = askopenfile(mode='r', filetypes=[('Python Files', '*.csv')])
    if file is not None:
        return file


def open_shipping_request_out():
    file = askopenfile(mode='r', filetypes=[('Python Files', '*.csv')])
    if file is not None:
        return file


def open_booking_request_inp():
    file = askopenfile(mode='r', filetypes=[('Python Files', '*.csv')])
    if file is not None:
        return file


def open_booking_request_out():
    file = askopenfile(mode='r', filetypes=[('Python Files', '*.csv')])
    if file is not None:
        return file


btn = Button(root, text='Select 1_SHIPPING_REQUEST_INP', command=lambda: open_shipping_request_inp())
btn.pack(side=TOP, pady=10)

btn = Button(root, text='Select 2_SHIPPING_REQUEST_OUT.csv', command=lambda: open_shipping_request_out())
btn.pack(side=TOP, pady=10)

btn = Button(root, text='Select 3_BOOKING_REQUEST_INP.csv', command=lambda: open_booking_request_inp())
btn.pack(side=TOP, pady=10)

btn = Button(root, text='Select 4_BOOKING_REQUEST_OUT.csv', command=lambda: open_booking_request_out())
btn.pack(side=TOP, pady=10)

B = Button(root, text="Hello", command=open_booking_request_out)

B.pack(side=TOP, pady=10)

root.mainloop()
