from flask import Flask, request, jsonify
from flask_cors import CORS
from functools import wraps
import cv2
import os
import numpy as np
import logging
import pymysql
import jwt
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename

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

def token_required(f):
    @wraps(f)
    def decorator(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token required'}), 401
        try:
            data = jwt.decode(token.split(" ")[1], 
                            app.config['SECRET_KEY'], 
                            algorithms=["HS256"])
            current_user = data['user_id']
        except:
            return jsonify({'message': 'Invalid token'}), 401
        return f(current_user, *args, **kwargs)
    return decorator

@app.route('/api/login', methods=['POST'])
def login():
    try:
        auth = request.get_json()
        if not auth or not auth.get('username') or not auth.get('password'):
            return jsonify({'message': 'Missing credentials'}), 400

        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE username = %s', 
                         (auth.get('username'),))
            user = cursor.fetchone()

            if not user or not check_password(auth.get('password'), user[2]):
                return jsonify({'message': 'Invalid credentials'}), 401

            token = jwt.encode({
                'user_id': user[0],
                'exp': datetime.utcnow() + timedelta(hours=24)
            }, app.config['SECRET_KEY'])

            return jsonify({'token': token}), 200

    except Exception as e:
        logging.error(f"Login error: {str(e)}")
        return jsonify({'message': 'Server error'}), 500

@app.route('/api/register-face', methods=['POST'])
@token_required
def register_face(current_user):
    try:
        if 'file' not in request.files:
            return jsonify({'message': 'No file provided'}), 400

        file = request.files['file']
        if not allowed_file(file.filename):
            return jsonify({'message': 'Invalid file type'}), 400

        # Process and save face data
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process face detection
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        img = cv2.imread(filepath)
        faces = face_cascade.detectMultiScale(img, 1.3, 5)

        if len(faces) == 0:
            os.remove(filepath)
            return jsonify({'message': 'No face detected'}), 400

        return jsonify({'message': 'Face registered successfully'}), 201

    except Exception as e:
        logging.error(f"Face registration error: {str(e)}")
        return jsonify({'message': 'Server error'}), 500

@app.route('/api/attendance', methods=['POST'])
@token_required
def mark_attendance(current_user):
    try:
        if 'file' not in request.files:
            return jsonify({'message': 'No file provided'}), 400

        file = request.files['file']
        # Process attendance marking logic here
        
        return jsonify({'message': 'Attendance marked successfully'}), 200

    except Exception as e:
        logging.error(f"Attendance marking error: {str(e)}")
        return jsonify({'message': 'Server error'}), 500

def get_db_connection():
    return pymysql.connect(**DB_CONFIG)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)