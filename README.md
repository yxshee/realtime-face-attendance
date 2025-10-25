# Realtime Face Attendance System

<img width="1164" alt="Face Attendance System" src="https://github.com/user-attachments/assets/7df7ab47-95e0-4cfa-b4fe-3bf1e00733c7">

A **highly efficient and accurate** real-time face attendance system using **OpenCV** and **LBPH Face Recognition**. Features both a desktop GUI application and a REST API for seamless attendance tracking.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Architecture](#architecture)
- [Demo](#demo)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

Attendance management is a critical component in educational institutions and organizations. This **Real-time Face Attendance** system provides an automated solution to streamline the attendance process using facial recognition technology. The system ensures accuracy, reduces manual effort, and enhances security by preventing proxy attendance.

## Features

### **Real-time Face Recognition**
Instantly recognize and record attendance as individuals enter the monitored area using live camera feeds.

### **Dual Interface**
- **Desktop Application**: Full-featured Tkinter GUI for local use
- **REST API**: Flask-based API for web integration

### **Multi-User Support**
Capable of handling multiple users simultaneously, ideal for classrooms or auditoriums.

### **User-Friendly Interface**
Intuitive dark-themed dashboard with four main tabs: Register, Train, Attendance, and Database.

### **Automated Report Generation**
Generates daily attendance CSV reports with timestamps for analysis and record-keeping.

## Architecture

<div align="center">

![Architecture Diagram](https://github.com/user-attachments/assets/234fe509-93e7-4655-ab9c-b9c1fadd3cee)

**System architecture showcasing the flow from image capture to attendance recording.**

</div>

### How It Works

1. **Image Capture**: Uses webcam to capture live video streams
2. **Face Detection**: Haar Cascade classifier detects faces in real-time
3. **Face Recognition**: LBPH (Local Binary Patterns Histograms) algorithm identifies individuals
4. **Attendance Logging**: Records recognized faces with timestamps to CSV files
5. **User Interface**: Displays real-time status and provides administrative controls

## Demo

<img width="878" alt="Demo Screenshot" src="https://github.com/user-attachments/assets/fe081adf-88cf-4c42-9609-b499b1f2b9c1" />

## Technologies Used

### Desktop Application (`codes/ultimate_system.py`)

| Technology | Purpose |
|------------|---------|
| **Python 3.8+** | Primary programming language |
| **OpenCV** | Real-time image/video processing |
| **opencv-contrib-python** | LBPH Face Recognizer |
| **Tkinter** | Desktop GUI framework |
| **NumPy** | Numerical operations |
| **Pandas** | CSV data handling |
| **Pillow** | Image processing for GUI |

### REST API (`deployment/api.py`)

| Technology | Purpose |
|------------|---------|
| **Flask** | Web framework |
| **Flask-CORS** | Cross-origin resource sharing |
| **MediaPipe** | Face detection |
| **PyJWT** | Authentication tokens |
| **PyMySQL** | MySQL database connector |
| **Gunicorn** | Production WSGI server |

## Project Structure

```
realtime-face-attendance/
├── codes/
│   └── ultimate_system.py      # Desktop Tkinter application (main)
├── deployment/
│   └── api.py                  # Flask REST API
├── model/
│   └── Haarcascade.xml         # Face detection model
├── database/
│   └── init_db.sql             # MySQL schema
├── docs/
│   └── PROJECT_DOCUMENTATION.md
├── TrainingImage/              # Captured face images (gitignored)
├── TrainingImageLabel/         # Trained model files (gitignored)
├── Attendance/                 # CSV attendance records (gitignored)
├── logs/                       # Application logs (gitignored)
├── .env.example                # Environment variables template
├── .gitignore
├── requirements.txt
├── LICENSE.md
└── README.md
```

## Installation

### Prerequisites

- **Python 3.8 or higher**
- **Git**
- **Webcam** for real-time video capture
- **MySQL** (optional, for API deployment)

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

4. **Configure Environment** (for API only)
    ```bash
    cp .env.example .env
    # Edit .env with your database credentials
    ```

## Usage

### Desktop Application (Recommended for Local Use)

```bash
# Activate virtual environment
source venv/bin/activate

# Run the desktop application
python codes/ultimate_system.py
```

#### Workflow

1. **Register Tab**: Enter student ID and name, then capture 60 face images
2. **Train Tab**: Train the LBPH model with captured images
3. **Attendance Tab**: Enter subject name and start real-time attendance tracking
4. **Database Tab**: View today's attendance records

### REST API (For Web Integration)

```bash
# Activate virtual environment
source venv/bin/activate

# Initialize database (MySQL required)
mysql -u root -p < database/init_db.sql

# Run the API server
python deployment/api.py
```

The API runs on `http://localhost:5001`

## API Documentation

### Endpoints

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/api/login` | Authenticate user | No |
| POST | `/api/register-face` | Register new face | Yes |
| POST | `/api/attendance` | Mark attendance | Yes |

### Authentication

All protected endpoints require a JWT token in the Authorization header:
```
Authorization: Bearer <your-token>
```

### Example: Login

```bash
curl -X POST http://localhost:5001/api/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "password"}'
```

## Deployment

### ⚠️ Important: Platform Considerations

This project has two deployment modes:

| Mode | Platform Recommendation | Use Case |
|------|------------------------|----------|
| Desktop App | Local machine | Classroom/office use |
| REST API | Railway, Render, Heroku | Web integration |

### Why Not Netlify?

Netlify is designed for static sites and serverless functions. This project requires:
- Continuous camera access (desktop only)
- Persistent server processes
- Real-time face recognition (exceeds serverless time limits)

### Recommended: Railway/Render Deployment (API Only)

1. Push code to GitHub
2. Connect repository to [Railway](https://railway.app) or [Render](https://render.com)
3. Set environment variables from `.env.example`
4. Deploy with start command: `gunicorn deployment.api:app`

### Procfile (for Heroku/Railway)

```
web: gunicorn deployment.api:app --bind 0.0.0.0:$PORT
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SECRET_KEY` | Flask secret key | Required |
| `DB_HOST` | MySQL host | localhost |
| `DB_USER` | MySQL username | root |
| `DB_PASSWORD` | MySQL password | Required |
| `DB_NAME` | Database name | face_attendance |

## Contributing

Contributions are welcome! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### How to Contribute

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/YourFeature`
3. Commit changes: `git commit -m "Add YourFeature"`
4. Push to branch: `git push origin feature/YourFeature`
5. Open a Pull Request

## License

This project is licensed under the [MIT License](LICENSE.md).

## Contact

- **Author**: [Yash Dogra](https://github.com/yxshee)
- **Email**: yxshdogra@gmail.com

## Acknowledgements

- **OpenCV** - Computer vision tools
- **MediaPipe** - Face detection
- **Python Community** - Excellent libraries and documentation
