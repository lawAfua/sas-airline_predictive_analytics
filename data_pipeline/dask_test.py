import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_values(value):
    try:
        return int(value)
    except Exception as e:
        print(e)
        return 0


from tkinter import *
from tkinter.ttk import *

from tkinter.filedialog import askopenfile

root = Tk()
root.title('Open SAS files for training')

root.geometry('500x200')

request_inp_file = ""
request_out_file = ""
booking_inp_file = ""
booking_out_file = ""


def open_shipping_request_inp():
    global request_inp_file
    file = askopenfile(mode='r', filetypes=[('Python Files', '*.csv')])
    if file is not None:
        request_inp_file = file.name
        return file


def open_shipping_request_out():
    global request_out_file
    file = askopenfile(mode='r', filetypes=[('Python Files', '*.csv')])
    if file is not None:
        request_out_file = file.name
        return file


def open_booking_request_inp():
    global booking_inp_file
    file = askopenfile(mode='r', filetypes=[('Python Files', '*.csv')])
    if file is not None:
        booking_inp_file = file.name
        return file


def open_booking_request_out():
    global booking_out_file
    file = askopenfile(mode='r', filetypes=[('Python Files', '*.csv')])
    if file is not None:
        booking_out_file = file.name
        return file


def load_file():
    btn = Button(root, text='Select 1_SHIPPING_REQUEST_INP', command=lambda: open_shipping_request_inp())
    btn.pack(side=TOP, pady=10)

    btn = Button(root, text='Select 2_SHIPPING_REQUEST_OUT.csv', command=lambda: open_shipping_request_out())
    btn.pack(side=TOP, pady=10)

    btn = Button(root, text='Select 3_BOOKING_REQUEST_INP.csv', command=lambda: open_booking_request_inp())
    btn.pack(side=TOP, pady=10)

    btn = Button(root, text='Select 4_BOOKING_REQUEST_OUT.csv', command=lambda: open_booking_request_out())
    btn.pack(side=TOP, pady=10)

    mainloop()


def get_final_data():
    # global request_inp_file
    load_file()
    print(request_inp_file)
    df = pd.read_csv(request_out_file, low_memory=False)

    df = df[df['SHIPRQ_COMPLETED'] == 'YES']

    df = df[['PRD_ID', 'PRD_CODE', 'WGT_CHRG_ACTUAL', 'CURRENCY', 'PRICE', 'TOT_BKFE', 'linkid', 'PI_PART_NAME']]

    df['linkid'] = df['linkid'].astype(int)
    df.drop_duplicates(['linkid'], inplace=True)

    df1 = pd.read_csv(request_inp_file, low_memory=False)[:64774]

    df2 = pd.read_csv(booking_inp_file)
    df2 = df2[df2['|REQUEST_COMPLETED|'] == '|YES|']

    df4 = pd.read_csv(booking_out_file)
    df4 = df4[df4['|REQUEST_COMPLETED|'] == '|YES|']

    df4 = df4[['|ORIGIN_STATION|', '|DEST_STATION|', '|ACTUAL_RATE|', '|TOT_CHARGE|', '|SUGGESTED_WGT|',
               '|SUGGESTED_VOL|', '|ORICAP_WGT|', '|ORICAP_VOL|', '|SEG_SPACE_PCS|', '|SEG_SPACE_WGT|',
               '|SEG_SPACE_VOL|', '|linkid|']]
    df2 = df2[['|PRD|', '|PARTINFO_NAME|', '|SUGGESTED_WUNI|', '|SUGGESTED_VUNI|',
               '|DESCRIPTION|', '|linkid|']]
    df2['linkid'] = df2['|linkid|'].dropna().apply(lambda x: x.strip('|'))
    df4['linkid'] = df4['|linkid|'].dropna().apply(lambda x: x.strip('|'))

    # df4.drop_duplicates(['linkid'], inplace=True)

    df1 = df1[['filedate', 'FIRST_ORIGIN', 'LAST_DEST', 'WUNI', 'VUNI', 'PIECES', 'WGT', 'VOL',
               'linkid']]
    df1['linkid'] = df1['linkid'].astype(int, errors='ignore')

    df2['linkid'] = df2['linkid'].dropna().apply(lambda x: get_values(x))
    df4['linkid'] = df4['linkid'].dropna().apply(lambda x: get_values(x))

    df3 = pd.merge(df2, df1, on='linkid', validate="many_to_many")
    print(df3)

    df3 = pd.merge(df3, df, on='linkid', validate="many_to_many")
    df3 = pd.merge(df3, df4, on='linkid', validate="many_to_many")

    df3 = df3.replace({'[|]': ''}, regex=True)
    return df3
    # df3.to_csv("merged_features11.csv", index=False)
    # print(df3.shape)

    # df3['|PRD|'] = df3['|PRD|'].dropna().apply(lambda x: x.strip('|'))
    # sns.countplot(x='|PRD|', data=df3)
    # plt.xticks(rotation=90)
    # plt.show()

    # df3['PRD_CODE'] = df3['PRD_CODE'].apply(lambda x: x.strip('|'))
    # sns.countplot(x='PRD_CODE', data=df3)
    # plt.xticks(rotation=90)
    # plt.show()

# get_final_data()
