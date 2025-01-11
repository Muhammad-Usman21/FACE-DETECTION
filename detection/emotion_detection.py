from deepface import DeepFace
from tensorflow.keras.models import load_model
import cv2

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

emotion_model = load_model("./models/facial_expression_model_weights.h5")
age_model = load_model("./models/age_model_weights.h5")
race_model = load_model(".models/race_model_single_batch.h5")

models = {
    'emotion': emotion_model,
    'age': age_model,
    'race': race_model,
}

def detect_emotions(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
        face_roi = rgb_frame[y:y + h, x:x + w]
        
        result = DeepFace.analyze(
            img_path=face_roi, 
            actions=['emotion', 'race', 'age'], 
            models=models,
            enforce_detection=False
        )
        
        emotion = result[0]['dominant_emotion']
        race = result[0]['dominant_race']
        age = result[0]['age']
        
        cv2.putText(
            img=frame,
            text=f"{emotion}, {race}, {age}",
            org=(x, y - 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2,
            color=(0, 0, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
            bottomLeftOrigin=False
        )
        cv2.rectangle(
            img=frame,
            pt1=(x, y),
            pt2=(x + w, y + h),
            color=(0, 0, 255),
            thickness=4,
            lineType=cv2.LINE_AA,
            shift=0
        )
        
    return frame
