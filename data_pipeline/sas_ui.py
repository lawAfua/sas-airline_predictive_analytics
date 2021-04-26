import time
from datetime import datetime

import PySimpleGUI as sg
import pandas as pd
from dateutil import parser
from predict_regression import get_regression_prediction
from statsmodels.tsa.arima_model import ARIMAResults
from regression_model import train_regression
from arima_sas import train_arima_sarima
# sg.theme('BluePurple')


def get_values(value):
    try:
        return int(value)
    except Exception as e:
        print(e)
        return 0


def get_final_data(request_inp_file, request_out_file, booking_inp_file, booking_out_file):
    print("Started preparing the training data")
    df = pd.read_csv(request_out_file, low_memory=False)

    df = df[df['SHIPRQ_COMPLETED'] == 'YES']

    df = df[['PRD_ID', 'PRD_CODE', 'WGT_CHRG_ACTUAL', 'CURRENCY', 'PRICE', 'TOT_BKFE', 'linkid', 'PI_PART_NAME']]

    df['linkid'] = df['linkid'].astype(int)
    # df.drop_duplicates(['linkid'], inplace=True)

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

    df3.to_csv("feature_file.csv")

    return df3


layout_1 = [[sg.T("")], [sg.Text("Select 1_SHIPPING_REQUEST_INP: "),
                         sg.Input(), sg.FileBrowse(key="1_SHIPPING_REQUEST_INP")],
            [sg.Text("Select 2_SHIPPING_REQUEST_OUT:"), sg.Input(), sg.FileBrowse(key="2_SHIPPING_REQUEST_OUT")],
            [sg.Text("Select 3_BOOKING_REQUEST_INP:  "), sg.Input(), sg.FileBrowse(key="3_BOOKING_REQUEST_INP")],
            [sg.Text("Select 4_BOOKING_REQUEST_OUT: "), sg.Input(), sg.FileBrowse(key="4_BOOKING_REQUEST_OUT")],
            [sg.Button("Start Merging and Training")], [sg.T(" " * 150), sg.Button('Exit')], [sg.Text("   "*50, key='train')]]

layout_2 = [[sg.T("")], [sg.Text('Predict future requests ', justification='center', size=(100, 1))],
            [sg.T("                                                         "), sg.In(key='-CAL-', enable_events=False,
                                                                                      visible=False),
             sg.CalendarButton('Select Date For Prediction', target='-CAL-',
                               pad=None, font=('MS Sans Serif',
                                               10, 'bold'),
                               key='_CALENDAR_',
                               format='%d %B, %Y')],
            [sg.Button("Start prediction", size=(95, 1))], [sg.Text('_' * 100)],
            [sg.Text("Prediction output", key="text1")], [sg.Multiline(size=(82, 10), key='Output1')],
            [sg.T(" " * 130), sg.Button('Clear', key='Clear1'), sg.Button('Exit')]]

layout_3 = [[sg.T("")], [sg.Text('Predict future requests ', justification='center', size=(100, 1))],
            [sg.T("   " * 15), sg.In(key='-CAL1-', enable_events=False,
                                     visible=False), sg.CalendarButton('From Date', target='-CAL1-',
                                                                       pad=None, font=('MS Sans Serif',
                                                                                       10, 'bold'),
                                                                       key='_CALENDAR1_',
                                                                       format='%d %B, %Y'),
             sg.In(key='-CAL2-', enable_events=False,
                   visible=False), sg.CalendarButton('To Date  ', target='-CAL2-',
                                                     pad=None, font=('MS Sans Serif',
                                                                     10, 'bold'),
                                                     key='_CALENDAR2_',
                                                     format='%d %B, %Y')
             ],
            [sg.Button("Start prediction ", size=(95, 1))], [sg.Text('_' * 100)],
            [sg.Text("Prediction output", key="text2")], [sg.Output(size=(82, 10), key='Output2')],
            [sg.T(" " * 130), sg.Button('Clear'), sg.Button('Exit')]]

layout = [[sg.TabGroup(
    [[sg.Tab('Merging and Training', layout_1), sg.Tab('Predict date wise', layout_2, key='tab2'), sg.Tab('Predict between date', layout_3, key='tab3')]],
    key='TABGROUP')]]

window = sg.Window('SAS Forecasting', layout, size=(700, 500), finalize=True)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == "Exit":
        break
    elif event == sg.WIN_CLOSED or event == "Exit0":
        break
    elif event == sg.WIN_CLOSED or event == "Exit1":
        break
    elif event == "Start Merging and Training":
        window['train'].update("DATA MERGING STARTED WILL TAKE LONG TIME BASED ON SYSTEM CONFIGURATION")
        # print("DATA MERGING STARTED WILL TAKE LONG TIME BASED ON SYSTEM CONFIGURATION")
        # time.sleep(5)

        df = get_final_data(values['1_SHIPPING_REQUEST_INP'], values['2_SHIPPING_REQUEST_OUT'],
                            values['3_BOOKING_REQUEST_INP'], values['4_BOOKING_REQUEST_OUT'])
        if df.shape[0]:
            print("ARIMA and SARIMA MODEL TRAINING STARTED.... IT WILL TAKE FEW MINUTES")
            window['train'].update("ARIMA, SARIMA AND REGRESSION MODEL TRAINING STARTED.... IT WILL TAKE FEW MINUTES")
            train_regression(df)
            train_arima_sarima(df)
            # time.sleep(5)
            window['train'].update("CONGRATULATIONS ARIMA, SARIMA AND REGRESSION MODEL TRAINING DONE")
    elif event == 'Start prediction':
        date_value = values.get('-CAL-')

        loaded = ARIMAResults.load('model_sarima.pkl')
        future = loaded.forecast(steps=36)

        current_time = datetime.now()
        validate_date = parser.parse(date_value) - parser.parse(str(current_time))
        validate_days = validate_date.days + 1
        if validate_date.days < 0:
            window['Output1'].update("Please select future date")
        else:
            start_date = future.index[-1]
            no_of_day = parser.parse(date_value) - start_date
            day = abs(no_of_day.days)
            loaded = ARIMAResults.load('model.pkl')
            future = loaded.forecast(steps=day + 1)[0]
            request_lst = ["ARIMA RESULT", str(abs(int(future[-1])))]
            window['Output1'].update('\n'.join(request_lst))

            loaded = ARIMAResults.load('model_sarima.pkl')
            future = loaded.forecast(steps=day + 1)

            request_lst.append('SARIMA RESULT')
            request_lst.append(str(abs(int(future.values[-1]))))

            window['Output1'].update('\n'.join(request_lst))

            request_lst.append("REGRESSION RESULT")
            df = pd.read_csv("feature_file.csv")
            result = get_regression_prediction(df, day)
            request_lst.append(str(result[0]))
            window['Output1'].update('\n'.join(request_lst))

    elif event == 'Clear':
        window['Output2'].update('')
    elif event == 'Clear1':
        window['Output1'].update('')
    elif event == 'Start prediction ':
        loaded = ARIMAResults.load('model_sarima.pkl')
        future = loaded.forecast(steps=36)

        current_time = datetime.now()
        validate_start_date = parser.parse(values.get('-CAL1-')) - parser.parse(str(current_time))
        validate_start_date = validate_start_date.days + 1
        validate_end_date = parser.parse(values.get('-CAL2-')) - parser.parse(str(current_time))
        validate_end_date = validate_end_date.days + 1

        if validate_start_date < 0 and validate_end_date < 0:
            print("Please select future date")
        else:
            loaded = ARIMAResults.load('model_sarima.pkl')
            future = loaded.forecast(steps=36)

            start_date = future.index[-1]
            no_of_day = parser.parse(values.get('-CAL2-')) - start_date
            day = abs(no_of_day.days)

            days = parser.parse(values.get('-CAL1-')) - parser.parse(values.get('-CAL2-'))
            # print(abs(days.days))
            latest_day = abs(days.days)+1
            # print(latest_day)
            print("Predicted Requests From {} To {}".format(values.get('-CAL1-'), values.get('-CAL2-')))
            print("ARIMA RESULT")
            loaded = ARIMAResults.load('model.pkl')
            future = loaded.forecast(steps=day + 1)[0]
            for req in future[day-(latest_day-1):]:
                print(abs(int(req)))
            print("SARIMA RESULT")
            loaded = ARIMAResults.load('model_sarima.pkl')
            future = loaded.forecast(steps=day + 1)
            for req in future.values[day-(latest_day-1):]:
                print(abs(int(req)))

            print("REGRESSION RESULT")
            df = pd.read_csv("feature_file.csv")
            result = get_regression_prediction(df, day)
            for req in result[day-latest_day:]:
                print(abs(int(req)))

window.close()
