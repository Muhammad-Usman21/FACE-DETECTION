
# Face Detection

This project uses Deep Learning and OpenCV to perform multiple image processing tasks such as face detection, mask detection, human emotion detection, and face count. It is implemented in Python using popular libraries such as PySimpleGUI, OpenCV, TensorFlow/Keras, and DeepFace.

------

## Features

1. **Face Detection**: Detect faces in images and draw bounding boxes around them.
2. **Face Count**: Count the number of faces detected in the image and display the count.
3. **Mask Detection**: Detect if a person is wearing a mask or not in the image.
4. **Human Emotion Detection**: Analyze the emotions of the detected faces (e.g., happy, sad, angry).
5. **Histogram Equalization**: Improve the image quality through histogram equalization to enhance contrast.
6. **GUI**: A simple graphical interface to interact with the program using PySimpleGUI.

-------

## Dependencies

To install the required dependencies, create a virtual environment and install the necessary libraries using the following:

```bash
pip install -r requirements.txt
```

Here are the main libraries used in the project:

- `OpenCV`: For image processing and face detection.
- `DeepFace`: For emotion, age, and race analysis.
- `PySimpleGUI`: For the graphical user interface.
- `TensorFlow`: For deep learning models used in mask and emotion detection.
- `PIL` (Pillow): For image manipulation and GUI integration.

------

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Muhammad-Usman21/FACE-DETECTION.git
```

2. Navigate to the project directory:

```bash
cd FACE-DETECTION
```

3. Set up a virtual environment to manage dependencies:

   - For Windows:
   
   ```bash
   python -m venv venv
   ```

   - For macOS/Linux:
   
   ```bash
   python3 -m venv venv
   ```

4. Activate the virtual environment:

   - For Windows:
   
   ```bash
   venv\Scripts\activate
   ```

   - For macOS/Linux:
   
   ```bash
   source venv/bin/activate
   ```

5. Install the required libraries:

```bash
pip install -r requirements.txt
```

6. Download the necessary pre-trained models for emotion detection, mask detection, and face recognition:
- `facial_expression_model_weights.h5` (Emotion Detection)
- `mask_recog.h5` (Mask Detection)
- `age_model_weights.h5` (Age Prediction)
- `race_model_single_batch.h5` (Race Prediction)
  
  Place these models in the `./models/` directory.

7. Run the application with the following command:

```bash
python main.py
```

-----

## Future Improvements

- Improve model accuracy and performance.
- Add support for webcam input.
- Integrate more image processing tasks (e.g., gender detection, facial landmark detection).
