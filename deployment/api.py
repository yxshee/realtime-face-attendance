# Standard library imports
import os
import logging
from datetime import datetime, timedelta
from functools import wraps

# Third-party imports
import cv2
import numpy as np
import pymysql
import jwt
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.security import check_password_hash
from werkzeug.utils import secure_filename
from werkzeug.exceptions import BadRequest, Unauthorized, InternalServerError

# Mediapipe imports
import mediapipe as mp

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-change-this')
app.config['UPLOAD_FOLDER'] = 'TrainingImage'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', 'root'),
    'db': os.getenv('DB_NAME', 'face_attendance'),
    'charset': 'utf8mb4'
}

# Setup logging
logging.basicConfig(
    filename='api.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# Token authentication decorator
def token_required(f):
    @wraps(f)
    def decorator(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token is missing'}), 401

        try:
            # Remove 'Bearer ' prefix if present
            if token.startswith('Bearer '):
                token = token.split(" ")[1]

            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = data['user_id']
        except ExpiredSignatureError:
            return jsonify({'message': 'Token has expired'}), 401
        except InvalidTokenError:
            return jsonify({'message': 'Invalid token'}), 401
        except Exception as e:
            logging.error(f"Token validation error: {str(e)}")
            return jsonify({'message': 'Token validation failed'}), 401

        return f(current_user, *args, **kwargs)
    return decorator


# API endpoint: Login
@app.route('/api/login', methods=['POST'])
def login():
    try:
        auth = request.get_json()
        if not auth or not auth.get('username') or not auth.get('password'):
            return jsonify({'message': 'Missing credentials'}), 400

        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE username = %s', (auth.get('username'),))
            user = cursor.fetchone()

            # user[0] -> user_id, user[1] -> username, user[2] -> hashed_password
            if not user or not check_password(auth.get('password'), user[2]):
                return jsonify({'message': 'Invalid credentials'}), 401

            # Generate JWT token
            token = jwt.encode({
                'user_id': user[0],
                'exp': datetime.utcnow() + timedelta(hours=24)
            }, app.config['SECRET_KEY'], algorithm='HS256')

            return jsonify({'token': token}), 200
    except Exception as e:
        logging.error(f"Login error: {str(e)}")
        return jsonify({'message': 'Server error'}), 500


# Face detection using Mediapipe
def detect_faces(image_path):
    """
    Detects faces in the given image using Mediapipe's Face Detection API.
    Returns True if faces are detected, otherwise False.
    """
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    image = cv2.imread(image_path)
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.detections:
            return True  # Face detected
    return False  # No face detected


# API endpoint: Register Face
@app.route('/api/register-face', methods=['POST'])
@token_required
def register_face(current_user):
    """
    API to register a face by detecting it from an uploaded image.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'message': 'No file provided'}), 400

        file = request.files['file']
        if not allowed_file(file.filename):
            return jsonify({'message': 'Invalid file type'}), 400

        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Face detection using Mediapipe
        face_detected = detect_faces(filepath)
        if not face_detected:
            os.remove(filepath)
            return jsonify({'message': 'No face detected'}), 400

        return jsonify({'message': 'Face registered successfully'}), 201
    except Exception as e:
        logging.error(f"Face registration error: {str(e)}")
        return jsonify({'message': 'Server error'}), 500


# API endpoint: Mark Attendance
@app.route('/api/attendance', methods=['POST'])
@token_required
def mark_attendance(current_user):
    """
    API to mark attendance based on an uploaded face image.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'message': 'No file provided'}), 400

        file = request.files['file']
        # Add attendance marking logic here

        return jsonify({'message': 'Attendance marked successfully'}), 200
    except Exception as e:
        logging.error(f"Attendance marking error: {str(e)}")
        return jsonify({'message': 'Server error'}), 500


# Database connection helper
def get_db_connection():
    """
    Establish a database connection using the configured credentials.
    """
    return pymysql.connect(**DB_CONFIG)


# File validation helper
def allowed_file(filename):
    """
    Validate that a file has an allowed extension.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# Register error handlers
@app.errorhandler(BadRequest)
@app.errorhandler(Unauthorized)
@app.errorhandler(InternalServerError)
def handle_error(e):
    logging.error(f"Error: {str(e)}")
    return jsonify({'message': str(e)}), getattr(e, 'code', 500)


# Run the app
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5001)
