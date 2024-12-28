# Real-time Face Attendance

<!--![License](https://img.shields.io/github/license/yxshee/realtime-face-attendance)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Stars](https://img.shields.io/github/stars/yxshee/realtime-face-attendance?style=social)
![Forks](https://img.shields.io/github/forks/yxshee/realtime-face-attendance?style=social)-->

A **highly efficient and accurate** real-time face attendance system leveraging **state-of-the-art computer vision** and **machine learning** technologies.Seamless and automated attendance tracking using facial recognition

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Architecture](#architecture)
- [Demo](#demo)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

## Introduction

Attendance management is a critical component in educational institutions and organizations. **Real-time Face Attendance** provides an automated solution to streamline the attendance process using advanced facial recognition technology. This system ensures accuracy, reduces manual effort, and enhances security by preventing proxy attendance.

## Features

### **Real-time Face Recognition**
Instantly recognize and record attendance as individuals enter the monitored area using live camera feeds.

### **High Accuracy**
Employs deep learning models trained on extensive datasets to ensure precise face recognition even in varying lighting and angles.

### **Multi-User Support**
Capable of handling multiple users simultaneously, making it ideal for large classrooms or auditoriums.

### **User-Friendly Interface**
Intuitive dashboard for easy management and monitoring of attendance records.

### **Secure Data Handling**
Ensures all attendance data is securely stored and processed, adhering to privacy standards.

### **Automated Report Generation**
Generates comprehensive attendance reports with customizable parameters for analysis and record-keeping.

## Architecture

<div align="center">
    
![Architecture Diagram](https://github.com/user-attachments/assets/234fe509-93e7-4655-ab9c-b9c1fadd3cee)

**System architecture showcasing the flow from image capture to attendance recording.**

 **Image Capture**: Utilizes webcams or IP cameras to capture live video streams.
 **Face Detection**: Processes frames using OpenCV to detect faces in real-time.
 **Face Recognition**: Applies a pre-trained deep learning model to identify individuals.
 **Attendance Logging**: Records recognized faces with timestamps in the database.
 **User Interface**: Displays real-time attendance status and provides administrative controls.

<img alt="Deepface Architecture by Facebook 
" src="https://github.com/user-attachments/assets/1b83650c-f85c-48c3-bfb3-d1bf16ba67fa" >
**Deepface Architecture by Facebook** </div>




## Demo

Experience the **Real-time Face Attendance** system in action!

### ![Demo Screenshot](https://github.com/yxshee/realtime-face-attendance/assets/demo-screenshot.png)

*Real-time face recognition interface displaying recognized users and attendance status.*

### Video Walkthrough

Watch a short video demonstrating the features and usage of Real-time Face Attendance:

[![Watch the video](https://img.icons8.com/ios-filled/50/000000/video.png)](https://www.youtube.com/watch?v=your-video-link)

## Technologies Used

- **Python 3.8+**
- **OpenCV:** For real-time image and video processing.
- **TensorFlow & Keras:** For building and deploying deep learning models.
- **dlib:** For robust face detection and landmark recognition.
- **SQLite/MySQL:** For managing attendance databases.
- **Flask/Django:** For creating the web-based user interface.
- **Bootstrap:** For responsive and modern UI design.
- **NumPy & Pandas:** For data manipulation and analysis.
- **Git & GitHub:** For version control and collaboration.

## Installation

### Prerequisites

- **Python 3.8 or higher**
- **Git**
- **Virtual Environment Tool** (e.g., `venv`, `conda`)
- **Webcam or IP Camera** for real-time video capture

### Steps

1. **Clone the Repository**
    ```bash
    git clone https://github.com/yxshee/realtime-face-attendance.git
    cd realtime-face-attendance
    ```

2. **Create a Virtual Environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Download Pre-trained Models**
    - Ensure that the `models/` directory contains the necessary pre-trained models. If not, follow the [Model Training](#model-training) section.

## Usage

### Running the Application

1. **Activate the Virtual Environment**
    ```bash
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2. **Launch the Application**
    ```bash
    python app.py
    ```

3. **Access the Interface**
    - Open your browser and navigate to `http://localhost:5000` to use the Real-time Face Attendance system.

### Using the Attendance System

1. **Register Users**
    - Add new users by uploading their facial images or capturing them via the camera.
    - Assign unique identifiers (e.g., student ID, employee ID) to each user.

    ![Register User](https://github.com/yxshee/realtime-face-attendance/assets/register-user.png)

2. **Start Attendance**
    - Click on the "Start Attendance" button to begin real-time face recognition.
    - The system will automatically detect and recognize faces, logging attendance in the database.

    ![Start Attendance](https://github.com/yxshee/realtime-face-attendance/assets/start-attendance.png)

3. **View Attendance Records**
    - Access the dashboard to view real-time attendance status and generate reports.

    ![Attendance Dashboard](https://github.com/yxshee/realtime-face-attendance/assets/attendance-dashboard.png)

4. **Generate Reports**
    - Export attendance data in various formats (e.g., CSV, PDF) for analysis and record-keeping.

    ![Generate Report](https://github.com/yxshee/realtime-face-attendance/assets/generate-report.png)

## Dataset

The **Real-time Face Attendance** system requires a dataset of user facial images for accurate recognition. You can create your own dataset by registering users through the application interface.

### Dataset Features

- **Diverse Users:** Supports multiple users with unique identifiers.
- **Varied Conditions:** Captures images under different lighting, angles, and expressions to enhance model robustness.
- **Secure Storage:** Ensures all facial data is securely stored and encrypted in the database.

> **Note:** For privacy and security reasons, the dataset is not publicly available. Ensure compliance with data protection regulations when collecting and storing facial data.

## Model Training

If you need to **train the face recognition model** from scratch or update it with new data, follow these steps:

### 1. Prepare the Dataset

Organize user facial images into the following directory structure:

```
data/
├── train/
│   ├── user1/
│   ├── user2/
│   └── userN/
├── validation/
│   ├── user1/
│   ├── user2/
│   └── userN/
```

### 2. Data Augmentation (Optional)

Enhance the dataset with augmented images to improve model robustness.

```bash
python data_augmentation.py --input_dir data/train/ --output_dir data/augmented_train/
```

### 3. Train the Model

```bash
python train_model.py --data_dir data/train/ --model_dir models/
```

### 4. Evaluate Performance

Assess the model's accuracy, precision, recall, and F1-score.

```bash
python evaluate_model.py --model_dir models/
```

### 5. Deploy the Model

Ensure the trained model files are placed in the `models/` directory for the application to use.

## Examples

### User Registration

![User Registration](https://github.com/yxshee/realtime-face-attendance/assets/user-registration.png)

*Registering a new user by capturing their facial images.*

### Real-time Attendance

![Real-time Attendance](https://github.com/yxshee/realtime-face-attendance/assets/real-time-attendance.png)

*System recognizing and logging attendance in real-time.*

### Attendance Reports

![Attendance Reports](https://github.com/yxshee/realtime-face-attendance/assets/attendance-reports.png)

*Comprehensive attendance reports generated by the system.*

## Contributing

Contributions are **welcome**! Whether it's reporting bugs, suggesting features, or submitting pull requests, your input helps improve the Real-time Face Attendance system.

### How to Contribute

1. **Fork the Repository**
    - Click the "Fork" button at the top-right corner of this page.

2. **Clone Your Fork**
    ```bash
    git clone https://github.com/your-username/realtime-face-attendance.git
    cd realtime-face-attendance
    ```

3. **Create a Feature Branch**
    ```bash
    git checkout -b feature/YourFeature
    ```

4. **Commit Your Changes**
    ```bash
    git commit -m "Add your feature"
    ```

5. **Push to the Branch**
    ```bash
    git push origin feature/YourFeature
    ```

6. **Open a Pull Request**
    - Navigate to the original repository and open a pull request from your fork.

### Contribution Guidelines

- **Code Quality:** Ensure your code follows the project's coding standards and is well-documented.
- **Testing:** Include relevant tests for new features or bug fixes.
- **Issue Tracking:** Before working on a new feature or bug, check existing issues to avoid duplicates.
- **Respect and Collaboration:** Be respectful and considerate in all interactions. Collaborate effectively with other contributors.

## License

This project is licensed under the [MIT License](LICENSE).

![MIT License](https://img.shields.io/github/license/yxshee/realtime-face-attendance)

## Contact

For any inquiries, issues, or contributions, please contact:

- **Author:** [Yash Dogra](https://github.com/yxshee)
- **Email:** yash999901@gmail.com

Feel free to open an issue or reach out directly for collaboration opportunities!

## Acknowledgements

- **OpenCV:** For providing powerful computer vision tools.
- **TensorFlow & Keras:** For enabling efficient deep learning model development.
- **dlib:** For robust face detection and landmark recognition.
- **Flask/Django:** For creating a seamless web-based user interface.
- **Bootstrap:** For responsive and modern UI design.
- **Community Contributors:** Special thanks to all contributors and supporters who helped make this project possible.

---


