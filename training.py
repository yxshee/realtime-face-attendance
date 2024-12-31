import cv2
import os
import numpy as np
from PIL import Image

# Initialize the face recognizer and cascade classifier
recognizer = cv2.face.FisherFaceRecognizer_create()
detector = cv2.CascadeClassifier('/Users/venom/Downloads/real_time_face_attendance/Haarcascade.xml')

def getImagesAndLabels(path):
    # Initialize lists
    faceSamples = []
    Ids = []
    current_id = 0
    label_dict = {}

    # Iterate through each image in the training directory
    for image_name in os.listdir(path):
        # Skip non-image files
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"Skipping non-image file: {image_name}")
            continue

        imagePath = os.path.join(path, image_name)
        print(f"Processing image: {imagePath}")

        # Extract person ID or name from the file name (assuming format: name.ID.jpg)
        try:
            # Example: "Yash.1.jpg" -> Person: "Yash", ID: 1
            person_name, person_id = image_name.rsplit('.', 2)[0:2]
            person_id = int(person_id)

            # Add person name and ID to label dictionary
            if person_name not in label_dict:
                label_dict[person_name] = person_id

            pilImage = Image.open(imagePath).convert('L')  # Convert to grayscale
            imageNp = np.array(pilImage, 'uint8')
            faces = detector.detectMultiScale(imageNp)

            if len(faces) == 0:
                print(f"No faces detected in {imagePath}")
                continue

            for (x, y, w, h) in faces:
                # Resize face to fixed size
                face_resized = cv2.resize(imageNp[y:y+h, x:x+w], (100, 100))
                faceSamples.append(face_resized)
                Ids.append(person_id)
        except Exception as e:
            print(f"Error processing image {imagePath}: {e}")

    if not label_dict:
        raise ValueError("No labels created. Ensure the images are named correctly.")

    print(f"Label mapping: {label_dict}")
    return faceSamples, Ids

# Get faces and IDs
training_path = '/Users/venom/Downloads/real_time_face_attendance/TrainingImage'
faces, Ids = getImagesAndLabels(training_path)

# Check the dimensions of faces and IDs
for i, face in enumerate(faces):
    print(f"Face {i} shape: {face.shape}")
    if face.shape != (100, 100):
        raise ValueError(f"Face at index {i} has invalid shape {face.shape}.")

faces = np.array(faces)
Ids = np.array(Ids, dtype=np.int32)

print(f"Faces array shape: {faces.shape}")
print(f"IDs array shape: {Ids.shape}")

if faces.size == 0 or Ids.size == 0:
    raise ValueError("No valid training data found. Please check the images and their formats.")

# Train the recognizer
recognizer.train(faces, Ids)
recognizer.save('/Users/venom/Downloads/real_time_face_attendance/TrainingImageLabel/trainner.yml')
print("Model trained and saved successfully.")

# Ensure the training script correctly saves Trainner.yml in the specified directory
