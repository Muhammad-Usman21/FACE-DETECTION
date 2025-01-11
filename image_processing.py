import cv2
import numpy as np
from PIL import Image, ImageTk
from detection.face_detection import detect_faces
from detection.mask_detection import detect_mask
from detection.emotion_detection import detect_emotions

def process_image(file_path, option):
    if file_path:
        color_image = cv2.imread(file_path)
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Histogram equalization
        height, width = gray_image.shape
        histogram = np.zeros(256, dtype=np.uint32)
        for y in range(height):
            for x in range(width):
                histogram[gray_image[y, x]] += 1

        total_pixels = np.sum(histogram)
        pdf = histogram / total_pixels
        cdf = np.cumsum(pdf)
        cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
        equalized_image = np.zeros_like(gray_image)

        for y in range(height):
            for x in range(width):
                equalized_image[y, x] = cdf_normalized[gray_image[y, x]]

        equalized_image = equalized_image.astype(np.uint8)
        frame = cv2.addWeighted(cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR), 0.5, color_image, 0.5, 0)

        if option == 'Face Detection':
            frame = detect_faces(frame)
        elif option == 'Mask Detection':
            frame = detect_mask(frame)
        elif option == 'Human Emotion Detection':
            frame = detect_emotions(frame)
        elif option == 'Face Count':
            frame, count = detect_faces(frame, count_faces=True)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img.thumbnail((800, 600))

        return ImageTk.PhotoImage(image=img)
