import cv2
print(cv2.__file__)

import os

# Load Haarcascade for face detection using OpenCV's built-in path
detector = cv2.CascadeClassifier('model/Haarcascade.xml')

# Verify that the Haar Cascade was loaded successfully
if detector.empty():
    raise IOError("Haar Cascade XML file not found or failed to load.")

# Path to a sample image
image_path = 'sys/Office.png'  # Update with your image path

# Read the image
image = cv2.imread(image_path)
if image is None:
    raise IOError(f"Cannot read image at {image_path}")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

print(f"Number of faces detected: {len(faces)}")

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image with detected faces
cv2.imshow('Face Detection Test', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Ensure face detection works correctly with the trained model and handles exceptions
