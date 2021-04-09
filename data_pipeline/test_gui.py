import PySimpleGUI as sg

layout = [[sg.T("")], [sg.Text('Predict future requests ', justification='center', size=(100, 1))],
          [sg.T("   "*10), sg.In(key='-CAL-', enable_events=False,
                                                    visible=False), sg.CalendarButton('Select Date From Prediction', target='-CAL-',
                                                                                     pad=None, font=('MS Sans Serif',
                                                                                                      10, 'bold'),
                                                                                     key='_CALENDAR_',
                                                                                     format='%d %B, %Y'),
           sg.In(key='-CAL1-', enable_events=False,
                 visible=False), sg.CalendarButton('Select Date To Prediction', target='-CAL1-',
                                                   pad=None, font=('MS Sans Serif',
                                                                   10, 'bold'),
                                                   key='_CALENDAR1_',
                                                   format='%d %B, %Y')
           ],
          [sg.Button("Start prediction", size=(95, 1))],  [sg.Text('_'*100)],
          [sg.Text("Prediction output", key="text1")], [sg.Output(size=(82, 10))],
          [sg.T(" "*150), sg.Button('Exit')]]

window = sg.Window('Predict future requests', layout, size=(700, 400), grab_anywhere=False)

while True:
    event, values = window.read()
    # print(event, values)
    print(values, 34534)
    if event in (None, 'Exit'):
        break
    # window["text1"].update(value="885989")

window.close()