import PySimpleGUI as sg

def create_window():
    layout = [
        [sg.Text("Digital Image Processing Project")],
        [sg.Radio('Face Detection', "RADIO1", default=True, key='Face Detection'),
         sg.Radio('Face Count', "RADIO1", key='Face Count'),
         sg.Radio('Mask Detection', "RADIO1", key='Mask Detection'),
         sg.Radio('Human Emotion Detection', "RADIO1", key='Human Emotion Detection')],
        [sg.Button("Open Image")],
        [sg.Image(key="-IMAGE-")]
    ]
    return sg.Window("DIP Project", layout)
