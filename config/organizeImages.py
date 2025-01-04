import os
import shutil

# Path to the main directory containing all images
image_dir = '/Users/venom/Downloads/real_time_face_attendance/TrainingImage'

def organize_images_into_subdirectories(image_dir):
    if not os.path.exists(image_dir):
        print(f"Error: The directory {image_dir} does not exist.")
        return

    # Iterate through all files in the directory
    for filename in os.listdir(image_dir):
        file_path = os.path.join(image_dir, filename)

        # Skip if it's not a file
        if not os.path.isfile(file_path):
            continue

        # Extract class name (use the part of the filename before the first dot)
        class_name = filename.split('.')[0]

        # Create a subdirectory for the class if it doesn't exist
        class_dir = os.path.join(image_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        # Move the file into the appropriate subdirectory
        try:
            shutil.move(file_path, os.path.join(class_dir, filename))
            print(f"Moved {filename} to {class_dir}")
        except Exception as e:
            print(f"Error moving {filename}: {e}")

# Ensure images are organized correctly for training
# Run the function
organize_images_into_subdirectories(image_dir)
