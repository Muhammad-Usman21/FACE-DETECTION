import PySimpleGUI as sg
from gui import create_window
from image_processing import process_image

def main():
    window = create_window()
    file_path = None
    prev_option = None

    while True:
        event, values = window.read(timeout=100)
        if event == sg.WINDOW_CLOSED:
            break
        elif event == "Open Image":
            file_path = sg.popup_get_file('Open', no_window=True)

        if file_path:
            option = [key for key, value in values.items() if value][0]
            if option != prev_option:
                prev_option = option
                img_tk = process_image(file_path, option)
                if img_tk:
                    window["-IMAGE-"].update(data=img_tk)

    window.close()

if __name__ == "__main__":
    main()
