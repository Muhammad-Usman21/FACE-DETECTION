import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model = load_model("./models/mask_recog.h5")
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

def detect_mask(frame):
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(
        image=gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(30, 30),
        maxSize=(200, 200)
    )

    for (x, y, w, h) in faces:
        face_frame = frame[y:y + h, x:x + w]
        face_frame = cv2.resize(face_frame, (224, 224))
        face_frame = img_to_array(face_frame)
        face_frame = preprocess_input(face_frame)
        face_frame = np.expand_dims(face_frame, axis=0)

        (mask, withoutMask) = model.predict(face_frame)[0]
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = f"{label}:\n{max(mask, withoutMask) * 100:.1f}%"
        lines = label.split('\n')

        for i, line in enumerate(lines):
            cv2.putText(
                img=frame,
                text=line,
                org=(x, y - 70 + (i * 50)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2,
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA,
                bottomLeftOrigin=False
            )
        cv2.rectangle(
            img=frame,
            pt1=(x, y),
            pt2=(x + w, y + h),
            color=color,
            thickness=4,
            lineType=cv2.LINE_AA,
            shift=0
        )

    return frame
