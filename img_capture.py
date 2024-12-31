import cv2
import os

# Path to save images
training_image_path = '/Users/venom/Downloads/real_time_face_attendance/TrainingImage'
if not os.path.exists(training_image_path):
    os.makedirs(training_image_path)
print(f"Images will be saved in: {training_image_path}")

# Webcam and face detector setup
cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier('/Users/venom/Downloads/real_time_face_attendance/Haarcascade.xml')

face_id = input('Enter a numeric ID for the person (e.g., 1, 2, etc.): ')
name = input('Enter the person\'s name: ')

print("Look at the camera and wait...")
count = 0

while True:
    ret, img = cam.read()
    if not ret:
        print("Error: Could not access the webcam.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        # Save face image
        file_path = f"{training_image_path}/{name}.{face_id}.{count}.jpg"
        cv2.imwrite(file_path, gray[y:y + h, x:x + w])
        print(f"Saved image at: {file_path}")
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Image Capture', img)

    # Stop after capturing 70 images or pressing 'q'
    if count >= 70 or cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

print(f"Training images saved for {name} with ID {face_id}.")
# Ensure images are captured and saved correctly for training
