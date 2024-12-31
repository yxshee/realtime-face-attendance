import cv2
import os
import numpy as np
from PIL import Image

# Path to Haar Cascade
cascadePath = '/Users/venom/Downloads/real_time_face_attendance/Haarcascade.xml'
faceCascade = cv2.CascadeClassifier(cascadePath)

# Path to Training Images
training_image_path = '/Users/venom/Downloads/real_time_face_attendance/TrainingImage'

model_save_path = '/Users/venom/Downloads/real_time_face_attendance/TrainingImageLabel/Trainner.yml'

# Create LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png')]
    face_samples = []
    ids = []

    for image_path in image_paths:
        pil_image = Image.open(image_path).convert('L')  # Convert to grayscale
        image_np = np.array(pil_image, 'uint8')
        face_id = int(os.path.split(image_path)[-1].split('.')[1])  # Extract ID from file name
        faces = faceCascade.detectMultiScale(image_np)
        
        for (x, y, w, h) in faces:
            face_samples.append(image_np[y:y + h, x:x + w])
            ids.append(face_id)

    return face_samples, ids

print("Training the model...")
faces, ids = get_images_and_labels(training_image_path)

if len(faces) == 0 or len(ids) == 0:
    print("No training images found. Please add images to the TrainingImage folder.")
    exit()

recognizer.train(faces, np.array(ids))
recognizer.write(model_save_path)  # Save the trained model
print(f"Model trained successfully and saved at {model_save_path}.")

# Ensure the testing script uses the correct database credentials and handles connection errors
