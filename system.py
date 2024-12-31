import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import csv
import os
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import pandas as pd
import datetime
import time
import pymysql
import threading
import subprocess
import platform
import hashlib
import json
import logging
import socket
from contextlib import contextmanager

# ===================== Configuration Section =====================

# Base Directory
BASE_DIR = "/Users/venom/Downloads/real_time_face_attendance"

# Subdirectories
STUDENT_DETAILS_DIR = os.path.join(BASE_DIR, "StudentDetails")
TRAINING_IMAGE_DIR = os.path.join(BASE_DIR, "TrainingImage")
TRAINING_IMAGE_LABEL_DIR = os.path.join(BASE_DIR, "TrainingImageLabel")
ATTENDANCE_DIR = os.path.join(BASE_DIR, "Attendance")
IMAGES_DIR = os.path.join(BASE_DIR, "images")  # Additional directory if needed

# Admin Credentials File
ADMIN_CREDENTIALS_FILE = os.path.join(BASE_DIR, "admin_credentials.json")

# Database Configuration
DB_HOST = 'localhost'
DB_USER = 'root'
DB_PASSWORD = 'root'  # Update if different
DB_NAME = 'Face_reco_fill'  # Ensure this database exists

# Haar Cascade Path
CASCADE_PATH = os.path.join(BASE_DIR, "Haarcascade.xml")

# Logging Configuration
LOG_FILE = os.path.join(BASE_DIR, 'app.log')
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# Additional Configuration
USE_DATABASE = True  # Set to False to disable database operations
DB_TIMEOUT = 3  # Connection timeout in seconds

# Add these color constants at the top after imports
THEME_COLORS = {
    'bg_dark': '#2D1B3D',        # Dark purple background
    'bg_medium': '#4C3A59',      # Medium purple
    'bg_light': '#6E5773',       # Light purple
    'accent': '#FF69B4',         # Pink accent
    'text_light': '#F4E1F5',     # Light text
    'text_dark': '#2D1B3D',      # Dark text
    'success': '#FF69B4',        # Success color (pink)
    'error': '#FF1493'           # Error color (deep pink)
}

# Add these constants after THEME_COLORS
NOTIFICATION_TYPES = {
    'success': {
        'bg': THEME_COLORS['bg_dark'],
        'fg': THEME_COLORS['success'],
        'duration': 5000,  # 5 seconds
        'prefix': '✓ Success: '
    },
    'error': {
        'bg': THEME_COLORS['bg_dark'],
        'fg': THEME_COLORS['error'],
        'duration': 8000,  # 8 seconds
        'prefix': '✗ Error: '
    },
    'info': {
        'bg': THEME_COLORS['bg_dark'],
        'fg': THEME_COLORS['accent'],
        'duration': 4000,  # 4 seconds
        'prefix': 'ℹ Info: '
    },
    'warning': {
        'bg': THEME_COLORS['bg_dark'],
        'fg': '#FFA500',  # Orange
        'duration': 6000,  # 6 seconds
        'prefix': '⚠ Warning: '
    }
}

class NotificationManager:
    def __init__(self, parent, max_notifications=3):
        self.parent = parent
        self.max_notifications = max_notifications
        self.notifications = []
        self.notification_frame = ttk.Frame(parent)
        self.notification_frame.grid(row=0, column=0, columnspan=3, sticky='EW')
        self.notification_frame.lift()  # Keep notifications on top

    def show(self, message, type='info'):
        config = NOTIFICATION_TYPES.get(type, NOTIFICATION_TYPES['info'])
        
        # Create notification label
        notification = ttk.Label(
            self.notification_frame,
            text=f"{config['prefix']}{message}",
            foreground=config['fg'],
            background=config['bg'],
            font=('Helvetica', 12, 'bold'),
            padding=(10, 5)
        )
        
        # Add to notifications list
        self.notifications.append({
            'label': notification,
            'created': time.time(),
            'duration': config['duration']
        })
        
        # Update display
        self._update_display()
        
        # Schedule cleanup
        self.parent.after(config['duration'], lambda: self._remove_notification(notification))

    def _update_display(self):
        # Remove old notifications if exceeding max
        while len(self.notifications) > self.max_notifications:
            oldest = self.notifications.pop(0)
            oldest['label'].destroy()
        
        # Update positions
        for i, notif in enumerate(self.notifications):
            notif['label'].grid(row=i, column=0, pady=(2, 2), sticky='EW')

    def _remove_notification(self, notification):
        for notif in self.notifications[:]:
            if notif['label'] == notification:
                self.notifications.remove(notif)
                notification.destroy()
                break
        self._update_display()

    def clear_all(self):
        for notif in self.notifications[:]:
            notif['label'].destroy()
        self.notifications.clear()

# ===================== Helper Functions =====================

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def initialize_admin_credentials():
    if not os.path.exists(ADMIN_CREDENTIALS_FILE):
        # Default credentials: username 'admin', password 'admin'
        default_credentials = {
            "admin": hash_password("admin")
        }
        with open(ADMIN_CREDENTIALS_FILE, 'w') as f:
            json.dump(default_credentials, f)
        logging.info("Initialized default admin credentials.")

def is_mysql_available():
    """Check if MySQL server is accessible"""
    try:
        socket.create_connection((DB_HOST, 3306), timeout=DB_TIMEOUT)
        return True
    except (socket.timeout, socket.error):
        return False

@contextmanager
def safe_cursor(show_error=True):
    """Context manager for safe database operations with fallback"""
    if not USE_DATABASE:
        yield None
        return
        
    try:
        connection = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            db=DB_NAME,
            connect_timeout=DB_TIMEOUT
        )
        with connection.cursor() as cursor:
            yield cursor
        connection.commit()
        connection.close()
    except Exception as e:
        if show_error:
            logging.error(f"Database Error: {str(e)}")
        yield None

# ===================== Application Class =====================

class FaceAttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Attendance System")
        
        # Make window responsive
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Set minimum window size
        self.root.minsize(1000, 700)
        self.root.geometry('1200x800')
        self.root.configure(background=THEME_COLORS['bg_dark'])

        # Update styling
        self.style = ttk.Style()
        try:
            self.style.theme_use('vista')
        except tk.TclError:
            self.style.theme_use('clam')

        # Configure styles with new theme colors
        self.style.configure('TButton',
                           font=('Helvetica', 12, 'bold'),
                           padding=8,
                           background=THEME_COLORS['accent'],
                           foreground=THEME_COLORS['text_light'])
        
        self.style.map('TButton',
                      background=[('active', THEME_COLORS['bg_light'])],
                      foreground=[('active', THEME_COLORS['text_light'])]
                      )

        self.style.configure('TLabel',
                           font=('Helvetica', 12),
                           background=THEME_COLORS['bg_dark'],
                           foreground=THEME_COLORS['text_light'])
        
        self.style.configure('Header.TLabel',
                           font=('Helvetica', 16, 'bold'),
                           background=THEME_COLORS['bg_dark'],
                           foreground=THEME_COLORS['accent'])

        self.style.configure('TEntry',
                           font=('Helvetica', 12),
                           fieldbackground=THEME_COLORS['bg_light'],
                           foreground=THEME_COLORS['text_light'])

        self.style.configure('TNotebook',
                           background=THEME_COLORS['bg_dark'],
                           tabmargins=[2, 5, 2, 0])
        
        self.style.configure('TNotebook.Tab',
                           background=THEME_COLORS['bg_medium'],
                           foreground=THEME_COLORS['text_light'],
                           padding=[10, 5],
                           font=('Helvetica', 10, 'bold'))
        
        self.style.map('TNotebook.Tab',
                      background=[('selected', THEME_COLORS['accent'])],
                      foreground=[('selected', THEME_COLORS['text_light'])])

        self.style.configure('Treeview',
                           background=THEME_COLORS['bg_medium'],
                           foreground=THEME_COLORS['text_light'],
                           fieldbackground=THEME_COLORS['bg_medium'],
                           bordercolor=THEME_COLORS['bg_light'],
                           borderwidth=1)
        
        self.style.configure('Treeview.Heading',
                           background=THEME_COLORS['bg_light'],
                           foreground=THEME_COLORS['text_light'],
                           font=('Helvetica', 12, 'bold'))
        
        self.style.map('Treeview',
                      background=[('selected', THEME_COLORS['accent'])],
                      foreground=[('selected', THEME_COLORS['text_light'])])

        # Update progress bar style
        self.style.configure('Horizontal.TProgressbar',
                           troughcolor=THEME_COLORS['bg_medium'],
                           background=THEME_COLORS['accent'],
                           bordercolor=THEME_COLORS['bg_light'],
                           lightcolor=THEME_COLORS['accent'],
                           darkcolor=THEME_COLORS['accent'])

        # Modify notification label colors
        self.notification_manager = NotificationManager(self.root)
        
        # Update notebook configuration
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=1, column=0, padx=20, pady=20, sticky='NSEW')

        # Configure grid weights for responsiveness
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)

        # Initialize Tabs
        self.init_registration_tab()
        self.init_training_tab()
        self.init_attendance_tab()
        self.init_admin_tab()

        # Check database availability
        if USE_DATABASE and not is_mysql_available():
            self.show_notification("Warning: Database server not accessible. System will operate in CSV-only mode.", fg=THEME_COLORS['error'])
            logging.warning("Database server not accessible at startup. Operating in CSV-only mode.")

    # ===================== Notification Methods =====================

    def on_tab_change(self, event):
        """Keep notifications visible when changing tabs"""
        # Don't clear notifications on tab change
        pass

    def show_notification(self, message, fg=None):
        """
        Show notification with proper color coding
        :param message: Message to display
        :param fg: Original color code or theme color key
        """
        # Map the old color codes to notification types
        type_mapping = {
            THEME_COLORS['error']: 'error',
            THEME_COLORS['success']: 'success',
            '#f85149': 'error',
            '#2ea44f': 'success'
        }
        
        notification_type = type_mapping.get(fg, 'info')
        self.notification_manager.show(message, type=notification_type)

    def clear_notification(self):
        self.notification_manager.clear_all()

    # Add this new method for smooth transitions
    def apply_hover_effects(self, widget):
        def on_enter(e):
            widget.configure(style='Hover.TButton')
        def on_leave(e):
            widget.configure(style='TButton')
        widget.bind('<Enter>', on_enter)
        widget.bind('<Leave>', on_leave)

    # ===================== Registration Tab =====================

    def init_registration_tab(self):
        self.registration_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.registration_tab, text='Register Student')

        # Labels and Entries
        ttk.Label(self.registration_tab, text="Enter Enrollment:", style='Header.TLabel').grid(row=0, column=0, padx=10, pady=10, sticky='W')
        self.enrollment_var = tk.StringVar()
        self.enrollment_entry = ttk.Entry(self.registration_tab, textvariable=self.enrollment_var, width=30)
        self.enrollment_entry.grid(row=0, column=1, padx=10, pady=10, sticky='W')

        ttk.Label(self.registration_tab, text="Enter Name:", style='Header.TLabel').grid(row=1, column=0, padx=10, pady=10, sticky='W')
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(self.registration_tab, textvariable=self.name_var, width=30)
        self.name_entry.grid(row=1, column=1, padx=10, pady=10, sticky='W')

        # Buttons
        self.clear_registration_btn = ttk.Button(self.registration_tab, text="Clear", command=self.clear_registration_fields)
        self.clear_registration_btn.grid(row=0, column=2, padx=10, pady=10)

        self.take_image_btn = ttk.Button(self.registration_tab, text="Take Images", command=self.start_take_images_thread)
        self.take_image_btn.grid(row=2, column=1, padx=10, pady=20, sticky='W')

        # Camera Display
        self.camera_frame = ttk.Frame(self.registration_tab, style='Camera.TFrame')
        self.camera_frame.grid(row=3, column=0, columnspan=3, padx=20, pady=20, sticky='NSEW')
        
        # Configure camera frame style
        self.style.configure('Camera.TFrame',
                           background=THEME_COLORS['bg_medium'],
                           borderwidth=2,
                           relief='solid')
        
        # Make camera display responsive
        self.camera_frame.grid_rowconfigure(0, weight=1)
        self.camera_frame.grid_columnconfigure(0, weight=1)
        
        self.camera_label = ttk.Label(self.camera_frame)
        self.camera_label.grid(row=0, column=0, sticky='NSEW')

        # Load and display logo in registration tab
        try:
            logo_path = os.path.join(BASE_DIR, "images", "logo.png")  # Add your logo file
            logo_image = Image.open(logo_path)
            logo_image = logo_image.resize((150, 150), Image.Resampling.LANCZOS)  # Adjust size as needed
            self.logo_photo = ImageTk.PhotoImage(logo_image)
            
            # Create and configure logo label in registration tab
            self.logo_label = ttk.Label(self.registration_tab, image=self.logo_photo)
            self.logo_label.grid(row=0, column=3, rowspan=2, padx=20, pady=20)
        except Exception as e:
            logging.warning(f"Could not load logo: {e}")

    def clear_registration_fields(self):
        self.enrollment_var.set("")
        self.name_var.set("")
        self.clear_notification()

    def start_take_images_thread(self):
        threading.Thread(target=self.take_img, daemon=True).start()

    def take_img(self):
        enrollment = self.enrollment_var.get().strip()
        name = self.name_var.get().strip()
        if not enrollment or not name:
            self.show_notification("Enrollment and Name are required!", fg=THEME_COLORS['error'])  # Red
            logging.warning("Attempted to take images without enrollment or name.")
            return

        self.show_notification("Starting image capture...", fg=THEME_COLORS['success'])  # Green
        logging.info(f"Starting image capture for Enrollment: {enrollment}, Name: {name}")

        try:
            cam = cv2.VideoCapture(0)
            if not cam.isOpened():
                self.show_notification("Cannot access the camera. Please ensure it's connected and not being used by another application.", fg=THEME_COLORS['error'])
                logging.error("Cannot access the camera.")
                return

            # Set camera resolution to HD
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            detector = cv2.CascadeClassifier(CASCADE_PATH)
            if detector.empty():
                self.show_notification("Failed to load Haar Cascade classifier. Please check the XML file.", fg=THEME_COLORS['error'])
                logging.error("Failed to load Haar Cascade classifier.")
                return

            sample_num = 0

            while True:
                ret, img = cam.read()
                if not ret:
                    self.show_notification("Failed to read from the camera.", fg=THEME_COLORS['error'])
                    logging.error("Failed to read from the camera.")
                    break
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (35, 134, 54), 2)  # Green rectangle
                    sample_num += 1
                    img_filename = f"{name}.{enrollment}.{sample_num}.jpg"
                    img_path = os.path.join(TRAINING_IMAGE_DIR, img_filename)
                    cv2.imwrite(img_path, gray[y:y + h, x:x + w])

                # Convert the image to RGB and then to PIL Image
                rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
                imgtk = ImageTk.PhotoImage(image=pil_image)

                # Schedule the GUI update in the main thread
                self.camera_label.after(0, lambda imgtk=imgtk: self.update_camera_feed(imgtk))

                if sample_num >= 70:
                    break

            cam.release()
            cv2.destroyAllWindows()

            # Save to CSV
            ts = time.time()
            Date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
            Time_Stamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            row = [enrollment, name, Date, Time_Stamp]

            csv_file_path = os.path.join(STUDENT_DETAILS_DIR, 'StudentDetails.csv')
            file_exists = os.path.isfile(csv_file_path)
            with open(csv_file_path, 'a', newline='') as csvFile:
                writer = csv.writer(csvFile)
                if not file_exists:
                    writer.writerow(['Enrollment', 'Name', 'Date', 'Time'])
                writer.writerow(row)

            self.show_notification(f"Images saved for Enrollment: {enrollment}, Name: {name}", fg=THEME_COLORS['success'])
            logging.info(f"Images saved for Enrollment: {enrollment}, Name: {name}")

        except cv2.error as cv_err:
            self.show_notification(f"OpenCV Error during image capture: {cv_err}", fg=THEME_COLORS['error'])
            logging.error("OpenCV Error during image capture.", exc_info=True)
        except Exception as e:
            self.show_notification(f"Error during image capture: {e}", fg=THEME_COLORS['error'])
            logging.error("Unexpected error during image capture.", exc_info=True)

    def update_camera_feed(self, imgtk):
        # Resize the image to be larger (zoomed)
        display_width = 1280  # Match attendance camera width
        display_height = 720  # Match attendance camera height
        
        # Resize the PIL image
        pil_image = imgtk._PhotoImage__photo.subsample(1)  # Remove subsampling
        pil_image = imgtk._PhotoImage__photo.zoom(1)  # Set zoom to 1x like attendance camera
        
        self.camera_label.imgtk = pil_image
        self.camera_label.configure(image=pil_image)

    # ===================== Training Tab =====================

    def init_training_tab(self):
        self.training_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.training_tab, text='Train Model')

        # Train Button
        self.train_model_btn = ttk.Button(self.training_tab, text="Train Images", command=self.start_train_model_thread)
        self.train_model_btn.grid(row=0, column=0, padx=20, pady=20, sticky='W')

        # Progress Bar
        self.train_progress = ttk.Progressbar(self.training_tab, orient='horizontal', length=400, mode='indeterminate')
        self.train_progress.grid(row=0, column=1, padx=20, pady=20, sticky='W')

    def start_train_model_thread(self):
        threading.Thread(target=self.trainimg, daemon=True).start()

    def trainimg(self):
        self.show_notification("Starting model training...", fg=THEME_COLORS['success'])  # Green
        logging.info("Model training started.")
        self.train_progress.start()
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        detector = cv2.CascadeClassifier(CASCADE_PATH)
        try:
            faces, Id = self.getImagesAndLabels(TRAINING_IMAGE_DIR, detector)
            if not faces or not Id:
                self.show_notification("No training images found. Please register students first.", fg=THEME_COLORS['error'])
                logging.warning("No training images found.")
                self.train_progress.stop()
                return
            recognizer.train(faces, np.array(Id))
            recognizer.save(os.path.join(TRAINING_IMAGE_LABEL_DIR, "Trainner.yml"))
            self.show_notification("Model trained successfully!", fg=THEME_COLORS['success'])
            logging.info("Model trained and saved successfully.")
        except cv2.error as cv_err:
            self.show_notification(f"OpenCV Error during training: {cv_err}", fg=THEME_COLORS['error'])
            logging.error("OpenCV Error during training.", exc_info=True)
        except Exception as e:
            self.show_notification(f"Error during training: {e}", fg=THEME_COLORS['error'])
            logging.error("Unexpected error during training.", exc_info=True)
        finally:
            self.train_progress.stop()

    def getImagesAndLabels(self, path, detector):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        faceSamples = []
        Ids = []
        for imagePath in imagePaths:
            try:
                pilImage = Image.open(imagePath).convert('L')  # grayscale
                imageNp = np.array(pilImage, 'uint8')
                # Extract Enrollment ID from filename
                parts = os.path.split(imagePath)[-1].split(".")
                if len(parts) < 3:
                    logging.warning(f"Skipping file {imagePath}: Incorrect naming convention.")
                    continue  # Skip files that don't follow the naming convention
                Enrollment = int(parts[1])
                faces = detector.detectMultiScale(imageNp)
                for (x, y, w, h) in faces:
                    faceSamples.append(imageNp[y:y + h, x:x + w])
                    Ids.append(Enrollment)
            except Exception as e:
                logging.warning(f"Skipping file {imagePath}: {e}")
        logging.debug(f"Collected {len(faceSamples)} faces for training.")
        return faceSamples, Ids

    # ===================== Attendance Tab =====================

    def init_attendance_tab(self):
        self.attendance_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.attendance_tab, text='Automatic Attendance')

        # Subject Entry
        ttk.Label(self.attendance_tab, text="Enter Subject:", style='Header.TLabel').grid(row=0, column=0, padx=10, pady=10, sticky='W')
        self.subject_var = tk.StringVar()
        self.subject_entry = ttk.Entry(self.attendance_tab, textvariable=self.subject_var, width=30)
        self.subject_entry.grid(row=0, column=1, padx=10, pady=10, sticky='W')

        # Buttons
        self.fill_attendance_btn = ttk.Button(self.attendance_tab, text="Fill Attendance", command=self.start_fill_attendance_thread)
        self.fill_attendance_btn.grid(row=1, column=0, padx=10, pady=20, sticky='W')

        self.check_sheets_btn = ttk.Button(self.attendance_tab, text="Check Sheets", command=self.open_attendance_folder)
        self.check_sheets_btn.grid(row=1, column=1, padx=10, pady=20, sticky='W')

        # Progress Bar
        self.attendance_progress = ttk.Progressbar(self.attendance_tab, orient='horizontal', length=400, mode='indeterminate')
        self.attendance_progress.grid(row=2, column=0, columnspan=2, padx=20, pady=10, sticky='W')

        # Camera Display for Attendance
        self.attendance_camera_frame = ttk.Frame(self.attendance_tab, style='Camera.TFrame')
        self.attendance_camera_frame.grid(row=3, column=0, columnspan=2, padx=20, pady=20, sticky='NSEW')
        
        # Make attendance camera display responsive
        self.attendance_camera_frame.grid_rowconfigure(0, weight=1)
        self.attendance_camera_frame.grid_columnconfigure(0, weight=1)
        
        self.attendance_camera_label = ttk.Label(self.attendance_camera_frame)
        self.attendance_camera_label.grid(row=0, column=0, sticky='NSEW')

    def start_fill_attendance_thread(self):
        threading.Thread(target=self.fill_attendance, daemon=True).start()

    def fill_attendance(self):
        subject = self.subject_var.get().strip()
        if not subject:
            self.show_notification("Please enter the subject name!", fg=THEME_COLORS['error'])  # Red
            logging.warning("Attempted to fill attendance without entering subject.")
            return

        self.show_notification("Starting attendance process...", fg=THEME_COLORS['success'])  # Green
        logging.info(f"Starting attendance process for subject: {subject}")
        self.attendance_progress.start()

        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            model_path = os.path.join(TRAINING_IMAGE_LABEL_DIR, "Trainner.yml")
            if not os.path.exists(model_path):
                self.show_notification("Model not found. Please train the model first.", fg=THEME_COLORS['error'])
                logging.error("Model file Trainner.yml not found.")
                self.attendance_progress.stop()
                return
            recognizer.read(model_path)
            faceCascade = cv2.CascadeClassifier(CASCADE_PATH)
            if faceCascade.empty():
                self.show_notification("Failed to load Haar Cascade classifier. Please check the XML file.", fg=THEME_COLORS['error'])
                logging.error("Failed to load Haar Cascade classifier.")
                self.attendance_progress.stop()
                return
            csv_file_path = os.path.join(STUDENT_DETAILS_DIR, "StudentDetails.csv")
            if not os.path.exists(csv_file_path):
                self.show_notification("StudentDetails.csv not found. Please register students first.", fg=THEME_COLORS['error'])
                logging.error("StudentDetails.csv not found.")
                self.attendance_progress.stop()
                return
            df = pd.read_csv(csv_file_path)
            cam = cv2.VideoCapture(0)
            if not cam.isOpened():
                self.show_notification("Cannot access the camera. Please ensure it's connected and not being used by another application.", fg=THEME_COLORS['error'])
                logging.error("Cannot access the camera for attendance.")
                self.attendance_progress.stop()
                return

            # Set camera resolution to HD
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            font = cv2.FONT_HERSHEY_SIMPLEX
            attendance_data = []

            ts = time.time()
            future = ts + 20  # 20 seconds for attendance

            while True:
                ret, im = cam.read()
                if not ret:
                    self.show_notification("Failed to access the camera.", fg=THEME_COLORS['error'])
                    logging.error("Failed to read from the camera during attendance.")
                    break
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale(gray, 1.2, 5)
                for (x, y, w, h) in faces:
                    Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
                    if conf < 70:
                        aa = df.loc[df['Enrollment'] == Id]['Name'].values
                        aa = aa[0] if len(aa) > 0 else "Unknown"
                        ts_current = time.time()
                        date = datetime.datetime.fromtimestamp(ts_current).strftime('%Y-%m-%d')
                        timeStamp = datetime.datetime.fromtimestamp(ts_current).strftime('%H:%M:%S')
                        attendance_data.append({'Enrollment': Id, 'Name': aa, 'Date': date, 'Time': timeStamp})
                        tt = f"{Id} - {aa}"
                        cv2.rectangle(im, (x, y), (x + w, y + h), (35, 134, 54), 2)  # Green rectangle
                        cv2.putText(im, tt, (x, y + h + 30), font, 1, (255, 255, 255), 2)
                    else:
                        cv2.rectangle(im, (x, y), (x + w, y + h), (245, 81, 73), 2)  # Red rectangle
                        cv2.putText(im, "Unknown", (x, y + h + 30), font, 1, (255, 255, 255), 2)

                # Convert the image to RGB and then to PIL Image
                rgb_image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
                imgtk = ImageTk.PhotoImage(image=pil_image)

                # Schedule the GUI update in the main thread
                self.attendance_camera_label.after(0, lambda imgtk=imgtk: self.update_attendance_camera_feed(imgtk))

                if cv2.waitKey(1) & 0xFF == ord('q') or time.time() > future:
                    break

            cam.release()
            cv2.destroyAllWindows()

            if not attendance_data:
                self.show_notification("No faces detected.", fg=THEME_COLORS['error'])
                logging.warning("No faces detected during attendance.")
                self.attendance_progress.stop()
                return

            # Create DataFrame
            attendance = pd.DataFrame(attendance_data).drop_duplicates(subset=['Enrollment'], keep='first')

            ts_final = time.time()
            date_final = datetime.datetime.fromtimestamp(ts_final).strftime('%Y-%m-%d')
            timeStamp_final = datetime.datetime.fromtimestamp(ts_final).strftime('%H:%M:%S')
            Hour, Minute, Second = timeStamp_final.split(":")
            fileName = f"{subject}_{date_final}_{Hour}-{Minute}-{Second}.csv"
            filePath = os.path.join(ATTENDANCE_DIR, fileName)
            attendance.to_csv(filePath, index=False)
            logging.info(f"Attendance recorded and saved to {filePath}")

            # Insert into Database
            if USE_DATABASE:
                if not is_mysql_available():
                    self.show_notification("Database server not accessible. Saving to CSV only.", fg=THEME_COLORS['error'])
                    logging.warning("Database server not accessible, proceeding with CSV only mode.")
                else:
                    DB_Table_name = f"{subject}_{date_final}_Time_{Hour}_{Minute}_{Second}"
                    with safe_cursor(show_error=True) as mycursor:
                        if mycursor:
                            try:
                                create_table_sql = f"""
                                    CREATE TABLE IF NOT EXISTS `{DB_Table_name}` (
                                        ID INT NOT NULL AUTO_INCREMENT,
                                        ENROLLMENT VARCHAR(100) NOT NULL,
                                        NAME VARCHAR(50) NOT NULL,
                                        DATE VARCHAR(20) NOT NULL,
                                        TIME VARCHAR(20) NOT NULL,
                                        PRIMARY KEY (ID)
                                    );
                                """
                                mycursor.execute(create_table_sql)
                                insert_sql = f"INSERT INTO `{DB_Table_name}` (ENROLLMENT, NAME, DATE, TIME) VALUES (%s, %s, %s, %s)"
                                for _, row in attendance.iterrows():
                                    mycursor.execute(insert_sql, (str(row['Enrollment']), row['Name'], row['Date'], row['Time']))
                                self.show_notification("Data saved to database and CSV!", fg=THEME_COLORS['success'])
                            except Exception as e:
                                self.show_notification(f"Database Error: {str(e)}. Data saved to CSV only.", fg=THEME_COLORS['error'])
                                logging.error(f"Database Error: {str(e)}")

            self.show_notification("Attendance filled successfully!", fg=THEME_COLORS['success'])
            logging.info("Attendance filled successfully.")

            # Open the attendance CSV
            self.open_file(filePath)

        except cv2.error as cv_err:
            self.show_notification(f"OpenCV Error during attendance: {cv_err}", fg=THEME_COLORS['error'])
            logging.error("OpenCV Error during attendance.", exc_info=True)
        except Exception as e:
            self.show_notification(f"Error during attendance: {e}", fg=THEME_COLORS['error'])
            logging.error("Unexpected error during attendance.", exc_info=True)
        finally:
            self.attendance_progress.stop()

    def update_attendance_camera_feed(self, imgtk):
        # Resize the image to be larger (zoomed)
        display_width = 1280 # Increased width
        display_height = 720  # Increased height
        
        # Resize the PIL image
        pil_image = imgtk._PhotoImage__photo.subsample(1)  # Remove subsampling
        pil_image = imgtk._PhotoImage__photo.zoom(1)  # Zoom level 2x
        
        self.attendance_camera_label.imgtk = pil_image
        self.attendance_camera_label.configure(image=pil_image)

    def open_attendance_folder(self):
        folder_path = ATTENDANCE_DIR
        if not os.path.exists(folder_path) or not os.listdir(folder_path):
            self.show_notification("No Attendance records found.", fg=THEME_COLORS['error'])
            logging.warning("Attempted to open attendance folder, but no records were found.")
            return
        self.open_file(folder_path)

    def open_file(self, path):
        try:
            if platform.system() == "Darwin":  # macOS
                subprocess.Popen(['open', path])
            elif platform.system() == "Windows":
                subprocess.Popen(['explorer', os.path.abspath(path)])
            elif platform.system() == "Linux":
                subprocess.Popen(['xdg-open', path])
            else:
                self.show_notification("Unsupported OS for opening files.", fg=THEME_COLORS['error'])
                logging.error("Unsupported OS for opening files.")
        except Exception as e:
            self.show_notification(f"Failed to open file/folder: {e}", fg=THEME_COLORS['error'])
            logging.error(f"Failed to open file/folder: {e}", exc_info=True)

    # ===================== Admin Tab =====================

    def init_admin_tab(self):
        self.admin_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.admin_tab, text='Admin Panel')

        # LogIn Labels and Entries
        ttk.Label(self.admin_tab, text="Username:", style='Header.TLabel').grid(row=0, column=0, padx=10, pady=10, sticky='W')
        self.admin_username_var = tk.StringVar()
        self.admin_username_entry = ttk.Entry(self.admin_tab, textvariable=self.admin_username_var, width=30)
        self.admin_username_entry.grid(row=0, column=1, padx=10, pady=10, sticky='W')

        ttk.Label(self.admin_tab, text="Password:", style='Header.TLabel').grid(row=1, column=0, padx=10, pady=10, sticky='W')
        self.admin_password_var = tk.StringVar()
        self.admin_password_entry = ttk.Entry(self.admin_tab, textvariable=self.admin_password_var, width=30, show="*")
        self.admin_password_entry.grid(row=1, column=1, padx=10, pady=10, sticky='W')

        # Buttons
        self.admin_login_btn = ttk.Button(self.admin_tab, text="LogIn", command=self.admin_login)
        self.admin_login_btn.grid(row=2, column=1, padx=10, pady=20, sticky='W')

    def admin_login(self):
        username = self.admin_username_var.get().strip()
        password = self.admin_password_var.get().strip()

        if not username or not password:
            self.show_notification("Please enter both username and password!", fg=THEME_COLORS['error'])
            logging.warning("Admin login attempted without username or password.")
            return

        try:
            with open(ADMIN_CREDENTIALS_FILE, 'r') as f:
                credentials = json.load(f)
        except FileNotFoundError:
            self.show_notification("Admin credentials file not found.", fg=THEME_COLORS['error'])
            logging.error("Admin credentials file not found.")
            return
        except json.JSONDecodeError:
            self.show_notification("Admin credentials file is corrupted.", fg=THEME_COLORS['error'])
            logging.error("Admin credentials file is corrupted.")
            return

        hashed_input_password = hash_password(password)
        stored_hashed_password = credentials.get(username)

        if stored_hashed_password and hashed_input_password == stored_hashed_password:
            self.show_notification("Admin logged in successfully!", fg=THEME_COLORS['success'])
            logging.info(f"Admin {username} logged in successfully.")
            self.display_students()
        else:
            self.show_notification("Incorrect Username or Password!", fg=THEME_COLORS['error'])
            logging.warning(f"Failed admin login attempt for username: {username}")

    def display_students(self):
        try:
            csv_file_path = os.path.join(STUDENT_DETAILS_DIR, "StudentDetails.csv")
            df = pd.read_csv(csv_file_path)
            logging.info("Loaded StudentDetails.csv successfully.")
        except FileNotFoundError:
            self.show_notification("StudentDetails.csv not found.", fg=THEME_COLORS['error'])
            logging.error("StudentDetails.csv not found.")
            return
        except Exception as e:
            self.show_notification(f"Error reading CSV: {e}", fg=THEME_COLORS['error'])
            logging.error(f"Error reading CSV: {e}", exc_info=True)
            return

        # Create a new window to display students
        students_window = tk.Toplevel(self.root)
        students_window.title("Registered Students")
        students_window.geometry('800x600')
        students_window.configure(background=THEME_COLORS['bg_dark'])  # Dark Background

        # Create a Treeview
        columns = ("Enrollment", "Name", "Date", "Time")
        tree = ttk.Treeview(students_window, columns=columns, show='headings')
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150)
        tree.pack(fill='both', expand=True)

        # Insert data into Treeview
        for _, row in df.iterrows():
            tree.insert('', tk.END, values=(row['Enrollment'], row['Name'], row['Date'], row['Time']))

        # Add a scrollbar
        scrollbar = ttk.Scrollbar(students_window, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Add a button to change admin password
        ttk.Button(students_window, text="Change Admin Password", command=self.change_admin_password).pack(pady=10)

    def change_admin_password(self):
        # Only accessible when admin is logged in
        change_pwd_window = tk.Toplevel(self.root)
        change_pwd_window.title("Change Admin Password")
        change_pwd_window.geometry('400x200')
        change_pwd_window.configure(background=THEME_COLORS['bg_dark'])  # Dark Background

        ttk.Label(change_pwd_window, text="Username:", style='Header.TLabel').grid(row=0, column=0, padx=10, pady=10, sticky='W')
        new_username_var = tk.StringVar()
        new_username_entry = ttk.Entry(change_pwd_window, textvariable=new_username_var, width=30)
        new_username_entry.grid(row=0, column=1, padx=10, pady=10, sticky='W')

        ttk.Label(change_pwd_window, text="New Password:", style='Header.TLabel').grid(row=1, column=0, padx=10, pady=10, sticky='W')
        new_password_var = tk.StringVar()
        new_password_entry = ttk.Entry(change_pwd_window, textvariable=new_password_var, width=30, show="*")
        new_password_entry.grid(row=1, column=1, padx=10, pady=10, sticky='W')

        def submit_new_password():
            username = new_username_var.get().strip()
            new_password = new_password_var.get().strip()

            if not username or not new_password:
                messagebox.showerror("Error", "Both fields are required.", parent=change_pwd_window)
                logging.warning("Attempted to change admin password without providing username or password.")
                return

            try:
                with open(ADMIN_CREDENTIALS_FILE, 'r') as f:
                    credentials = json.load(f)
            except FileNotFoundError:
                credentials = {}
                logging.warning("Admin credentials file not found while changing password. Creating a new one.")
            except json.JSONDecodeError:
                messagebox.showerror("Error", "Admin credentials file is corrupted.", parent=change_pwd_window)
                logging.error("Admin credentials file is corrupted while attempting to change password.")
                return

            credentials[username] = hash_password(new_password)

            with open(ADMIN_CREDENTIALS_FILE, 'w') as f:
                json.dump(credentials, f)

            messagebox.showinfo("Success", "Admin password updated successfully.", parent=change_pwd_window)
            logging.info(f"Admin password updated for username: {username}")
            change_pwd_window.destroy()

        ttk.Button(change_pwd_window, text="Submit", command=submit_new_password).grid(row=2, column=1, padx=10, pady=20, sticky='E')

# ===================== Main Execution =====================

def main():
    initialize_admin_credentials()  # Initialize admin credentials
    root = tk.Tk()
    app = FaceAttendanceApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: on_closing(root))
    root.mainloop()

def on_closing(root):
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        logging.info("Application closed by user.")
        root.destroy()

if __name__ == "__main__":
    main()
