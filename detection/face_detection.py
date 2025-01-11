import cv2

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

def detect_faces(frame, count_faces=False):
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(
        image=gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(30, 30),
        maxSize=(200, 200)
    )

    if count_faces:
        face_count = len(faces)
        
        cv2.putText(
            img=frame,
            text=f'Face Count: {face_count}',
            org=(frame.shape[1] // 10, frame.shape[0] // 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=5,
            color=(0, 255, 0),
            thickness=3,
            lineType=cv2.LINE_AA,
            bottomLeftOrigin=False
        )

    for (x, y, w, h) in faces:        
        cv2.rectangle(
            img=frame,
            pt1=(x, y),
            pt2=(x + w, y + h),
            color=(0, 255, 0),
            thickness=4,
            lineType=cv2.LINE_AA,
            shift=0
        )

    if count_faces:
        return frame, face_count
    return frame
