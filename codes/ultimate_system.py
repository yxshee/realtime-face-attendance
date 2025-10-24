#!/usr/bin/env python3
"""
Face Attendance System
Clean • Minimal • Reliable
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import threading
from datetime import datetime
import platform
import subprocess

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DIRS = {
    'training': os.path.join(BASE_DIR, "TrainingImage"),
    'models': os.path.join(BASE_DIR, "TrainingImageLabel"),
    'attendance': os.path.join(BASE_DIR, "Attendance"),
}

for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

CASCADE_PATH = os.path.join(BASE_DIR, "model", "Haarcascade.xml")

# ═══════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Attendance")
        self.root.geometry("1100x750")
        self.root.configure(bg='#000000')
        self.root.minsize(900, 600)
        
        # Minimal cursor (crosshair is cleaner than default)
        self.root.config(cursor='crosshair')
        
        # State
        self.camera = None
        self.running = False
        self.face_cascade = None
        self.recognizer = None
        self.current_frame = None
        self.current_tab = 0
        
        # Load models
        self._load_models()
        
        # Build UI
        self._build_ui()
        
        # Center window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() - 1100) // 2
        y = (self.root.winfo_screenheight() - 750) // 2
        self.root.geometry(f"+{x}+{y}")
    
    def _load_models(self):
        """Load face detection and recognition models"""
        try:
            if os.path.exists(CASCADE_PATH):
                self.face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
            else:
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            model_path = os.path.join(DIRS['models'], "Trainner.yml")
            if os.path.exists(model_path):
                try:
                    self.recognizer = cv2.face.LBPHFaceRecognizer_create()
                    self.recognizer.read(model_path)
                except:
                    pass
        except Exception as e:
            print(f"Model error: {e}")
    
    def _build_ui(self):
        """Build the minimal UI"""
        # ─── Header ───
        header = tk.Frame(self.root, bg='#000000')
        header.pack(fill='x', padx=40, pady=(30, 20))
        
        tk.Label(header, text="FACE ATTENDANCE", bg='#000000', fg='#ffffff',
                font=('Helvetica', 24, 'bold')).pack(side='left')
        
        self.status_label = tk.Label(header, text="Ready", bg='#000000', 
                                     fg='#666666', font=('Helvetica', 11))
        self.status_label.pack(side='right')
        
        # ─── Tab bar ───
        tab_bar = tk.Frame(self.root, bg='#000000')
        tab_bar.pack(fill='x', padx=40)
        
        self.tab_buttons = []
        tabs = [("Register", 0), ("Train", 1), ("Attendance", 2), ("Database", 3)]
        
        for text, idx in tabs:
            btn = tk.Button(tab_bar, text=text, bg='#000000', fg='#666666',
                           font=('Helvetica', 12), bd=0,
                           activebackground='#ffffff', activeforeground='#000000',
                           highlightthickness=0, padx=20, pady=10,
                           command=lambda i=idx: self._switch_tab(i))
            btn.pack(side='left')
            self.tab_buttons.append(btn)
        
        # Separator
        tk.Frame(self.root, bg='#333333', height=1).pack(fill='x', padx=40)
        
        # ─── Content container ───
        self.content = tk.Frame(self.root, bg='#000000')
        self.content.pack(fill='both', expand=True, padx=40, pady=20)
        
        # Create all tabs
        self.tabs = []
        self._create_register_tab()
        self._create_train_tab()
        self._create_attendance_tab()
        self._create_database_tab()
        
        # Show first tab
        self._switch_tab(0)
    
    def _switch_tab(self, idx):
        """Switch to tab by index"""
        self.current_tab = idx
        
        # Update tab buttons - FIXED: black text on white bg when selected
        for i, btn in enumerate(self.tab_buttons):
            if i == idx:
                # Selected: white background, black text
                btn.config(bg='#ffffff', fg='#000000', font=('Helvetica', 12, 'bold'))
            else:
                # Not selected: black background, gray text
                btn.config(bg='#000000', fg='#666666', font=('Helvetica', 12))
        
        # Show/hide tabs
        for i, tab in enumerate(self.tabs):
            if i == idx:
                tab.pack(fill='both', expand=True)
            else:
                tab.pack_forget()
        
        # Stop camera if switching away
        if self.running:
            self.running = False
            if self.camera:
                self.camera.release()
                self.camera = None
        
        # Refresh database if switching to database tab
        if idx == 3:
            self._refresh_database()
    
    def _create_register_tab(self):
        """Create registration tab"""
        tab = tk.Frame(self.content, bg='#000000')
        self.tabs.append(tab)
        
        # Two columns
        left = tk.Frame(tab, bg='#0a0a0a', padx=30, pady=30)
        left.pack(side='left', fill='y')
        
        right = tk.Frame(tab, bg='#000000')
        right.pack(side='right', fill='both', expand=True, padx=(20, 0))
        
        # ─── Form ───
        tk.Label(left, text="NEW STUDENT", bg='#0a0a0a', fg='#ffffff',
                font=('Helvetica', 14, 'bold')).pack(anchor='w', pady=(0, 25))
        
        # ID field
        tk.Label(left, text="ID", bg='#0a0a0a', fg='#666666',
                font=('Helvetica', 10)).pack(anchor='w')
        self.id_entry = tk.Entry(left, bg='#1a1a1a', fg='#ffffff',
                                font=('Helvetica', 12), bd=0,
                                insertbackground='#ffffff', width=25)
        self.id_entry.pack(fill='x', pady=(5, 15), ipady=10)
        
        # Name field
        tk.Label(left, text="NAME", bg='#0a0a0a', fg='#666666',
                font=('Helvetica', 10)).pack(anchor='w')
        self.name_entry = tk.Entry(left, bg='#1a1a1a', fg='#ffffff',
                                  font=('Helvetica', 12), bd=0,
                                  insertbackground='#ffffff', width=25)
        self.name_entry.pack(fill='x', pady=(5, 25), ipady=10)
        
        # Buttons
        btn_frame = tk.Frame(left, bg='#0a0a0a')
        btn_frame.pack(fill='x')
        
        tk.Button(btn_frame, text="● CAPTURE", bg='#ffffff', fg='#000000',
                 font=('Helvetica', 11, 'bold'), bd=0, padx=20, pady=10,
                 activebackground='#cccccc', activeforeground='#000000',
                 command=self._start_capture).pack(side='left', padx=(0, 10))
        
        tk.Button(btn_frame, text="CLEAR", bg='#333333', fg='#ffffff',
                 font=('Helvetica', 11), bd=0, padx=15, pady=10,
                 activebackground='#444444', activeforeground='#ffffff',
                 command=self._clear_form).pack(side='left')
        
        # Capture status
        self.capture_label = tk.Label(left, text="", bg='#0a0a0a', 
                                      fg='#666666', font=('Helvetica', 10))
        self.capture_label.pack(anchor='w', pady=(20, 0))
        
        # ─── Camera ───
        self.reg_camera = tk.Label(right, bg='#0a0a0a', text="Camera",
                                   fg='#333333', font=('Helvetica', 14))
        self.reg_camera.pack(expand=True, fill='both')
    
    def _create_train_tab(self):
        """Create training tab"""
        tab = tk.Frame(self.content, bg='#000000')
        self.tabs.append(tab)
        
        # Center content
        center = tk.Frame(tab, bg='#0a0a0a', padx=60, pady=50)
        center.place(relx=0.5, rely=0.5, anchor='center')
        
        tk.Label(center, text="◉", bg='#0a0a0a', fg='#ffffff',
                font=('Helvetica', 48)).pack(pady=(0, 20))
        
        tk.Label(center, text="TRAIN MODEL", bg='#0a0a0a', fg='#ffffff',
                font=('Helvetica', 16, 'bold')).pack(pady=(0, 10))
        
        tk.Label(center, text="Process captured images to train recognition",
                bg='#0a0a0a', fg='#666666', font=('Helvetica', 11)).pack(pady=(0, 30))
        
        tk.Button(center, text="START TRAINING", bg='#ffffff', fg='#000000',
                 font=('Helvetica', 12, 'bold'), bd=0, padx=30, pady=12,
                 activebackground='#cccccc', activeforeground='#000000',
                 command=self._start_training).pack()
        
        self.train_label = tk.Label(center, text="", bg='#0a0a0a',
                                   fg='#666666', font=('Helvetica', 10))
        self.train_label.pack(pady=(20, 0))
    
    def _create_attendance_tab(self):
        """Create attendance tab"""
        tab = tk.Frame(self.content, bg='#000000')
        self.tabs.append(tab)
        
        # Top bar
        top = tk.Frame(tab, bg='#0a0a0a', padx=20, pady=15)
        top.pack(fill='x')
        
        tk.Label(top, text="SUBJECT", bg='#0a0a0a', fg='#666666',
                font=('Helvetica', 10)).pack(side='left')
        
        self.subject_entry = tk.Entry(top, bg='#1a1a1a', fg='#ffffff',
                                     font=('Helvetica', 11), bd=0,
                                     insertbackground='#ffffff', width=20)
        self.subject_entry.pack(side='left', padx=(10, 30), ipady=8)
        
        tk.Button(top, text="▶ START", bg='#ffffff', fg='#000000',
                 font=('Helvetica', 10, 'bold'), bd=0, padx=15, pady=8,
                 activebackground='#cccccc', activeforeground='#000000',
                 command=self._start_attendance).pack(side='left', padx=(0, 10))
        
        tk.Button(top, text="■ STOP", bg='#333333', fg='#ffffff',
                 font=('Helvetica', 10), bd=0, padx=15, pady=8,
                 activebackground='#444444', activeforeground='#ffffff',
                 command=self._stop_attendance).pack(side='left', padx=(0, 10))
        
        tk.Button(top, text="REPORTS", bg='#1a1a1a', fg='#ffffff',
                 font=('Helvetica', 10), bd=0, padx=15, pady=8,
                 activebackground='#333333', activeforeground='#ffffff',
                 command=self._open_reports).pack(side='right')
        
        # Camera
        self.att_camera = tk.Label(tab, bg='#0a0a0a', text="Camera",
                                   fg='#333333', font=('Helvetica', 14))
        self.att_camera.pack(expand=True, fill='both', pady=(20, 0))
        
        # Status
        self.att_status = tk.Label(tab, text="", bg='#000000',
                                  fg='#666666', font=('Helvetica', 10))
        self.att_status.pack(pady=(10, 0))
    
    def _create_database_tab(self):
        """Create database viewer tab"""
        tab = tk.Frame(self.content, bg='#000000')
        self.tabs.append(tab)
        
        # Header
        header = tk.Frame(tab, bg='#0a0a0a', padx=20, pady=15)
        header.pack(fill='x')
        
        tk.Label(header, text="TODAY'S ATTENDANCE", bg='#0a0a0a', fg='#ffffff',
                font=('Helvetica', 14, 'bold')).pack(side='left')
        
        tk.Button(header, text="↻ REFRESH", bg='#333333', fg='#ffffff',
                 font=('Helvetica', 10), bd=0, padx=15, pady=8,
                 activebackground='#444444', activeforeground='#ffffff',
                 command=self._refresh_database).pack(side='right')
        
        # Table container with scrollbar
        table_frame = tk.Frame(tab, bg='#0a0a0a')
        table_frame.pack(fill='both', expand=True, pady=(20, 0))
        
        # Create canvas for scrolling
        self.db_canvas = tk.Canvas(table_frame, bg='#0a0a0a', highlightthickness=0)
        scrollbar = tk.Scrollbar(table_frame, orient='vertical', command=self.db_canvas.yview)
        self.db_scroll_frame = tk.Frame(self.db_canvas, bg='#0a0a0a')
        
        self.db_scroll_frame.bind('<Configure>', 
            lambda e: self.db_canvas.configure(scrollregion=self.db_canvas.bbox('all')))
        
        self.db_canvas.create_window((0, 0), window=self.db_scroll_frame, anchor='nw')
        self.db_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.db_canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Table header
        self.db_header = tk.Frame(self.db_scroll_frame, bg='#1a1a1a')
        self.db_header.pack(fill='x', padx=10, pady=(10, 5))
        
        headers = ['ID', 'Date', 'Time', 'Subject', 'Status']
        widths = [10, 12, 10, 20, 10]
        for h, w in zip(headers, widths):
            tk.Label(self.db_header, text=h, bg='#1a1a1a', fg='#ffffff',
                    font=('Helvetica', 10, 'bold'), width=w, anchor='w').pack(side='left', padx=5)
        
        # Data container
        self.db_data_frame = tk.Frame(self.db_scroll_frame, bg='#0a0a0a')
        self.db_data_frame.pack(fill='both', expand=True, padx=10)
        
        # Status label
        self.db_status = tk.Label(tab, text="", bg='#000000',
                                 fg='#666666', font=('Helvetica', 10))
        self.db_status.pack(pady=(10, 0))
    
    def _refresh_database(self):
        """Refresh database view with today's attendance"""
        # Clear existing data
        for widget in self.db_data_frame.winfo_children():
            widget.destroy()
        
        # Get today's file
        today = datetime.now().strftime('%Y-%m-%d')
        filename = f"Attendance_{today}.csv"
        filepath = os.path.join(DIRS['attendance'], filename)
        
        if not os.path.exists(filepath):
            self.db_status.config(text="No attendance records for today")
            return
        
        try:
            df = pd.read_csv(filepath)
            
            if df.empty:
                self.db_status.config(text="No records found")
                return
            
            # Add rows
            for idx, row in df.iterrows():
                row_frame = tk.Frame(self.db_data_frame, bg='#0a0a0a')
                row_frame.pack(fill='x', pady=2)
                
                # Alternate row colors
                row_bg = '#0a0a0a' if idx % 2 == 0 else '#111111'
                row_frame.config(bg=row_bg)
                
                values = [
                    str(row.get('ID', '')),
                    str(row.get('Date', '')),
                    str(row.get('Time', '')),
                    str(row.get('Subject', '')),
                    str(row.get('Status', ''))
                ]
                widths = [10, 12, 10, 20, 10]
                
                for val, w in zip(values, widths):
                    tk.Label(row_frame, text=val, bg=row_bg, fg='#cccccc',
                            font=('Helvetica', 10), width=w, anchor='w').pack(side='left', padx=5, pady=5)
            
            self.db_status.config(text=f"Loaded {len(df)} records")
            
        except Exception as e:
            self.db_status.config(text=f"Error: {e}")
    
    # ═══════════════════════════════════════════════════════════
    # FUNCTIONALITY
    # ═══════════════════════════════════════════════════════════
    
    def _status(self, msg):
        """Update status"""
        self.status_label.config(text=msg)
    
    def _clear_form(self):
        """Clear registration form"""
        self.id_entry.delete(0, 'end')
        self.name_entry.delete(0, 'end')
        self.capture_label.config(text="")
        self._status("Cleared")
    
    def _start_capture(self):
        """Start image capture"""
        student_id = self.id_entry.get().strip()
        name = self.name_entry.get().strip()
        
        if not student_id or not name:
            self._status("Fill all fields")
            return
        
        self.running = True
        self._status("Capturing...")
        threading.Thread(target=self._capture_loop, args=(student_id, name), daemon=True).start()
    
    def _capture_loop(self, student_id, name):
        """Capture images in background"""
        try:
            # Try to open camera
            self.camera = cv2.VideoCapture(0)
            
            if not self.camera.isOpened():
                for i in [1, 2, -1]:
                    self.camera = cv2.VideoCapture(i)
                    if self.camera.isOpened():
                        break
            
            if not self.camera.isOpened():
                self.root.after(0, lambda: self._status("Camera not found"))
                return
            
            # Camera settings
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            count = 0
            max_count = 60
            
            while self.running and count < max_count:
                ret, frame = self.camera.read()
                if not ret:
                    continue
                
                # Convert for display (FULL FRAME, not cropped)
                display_frame = frame.copy()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if self.face_cascade is not None:
                    faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                    
                    for (x, y, w, h) in faces:
                        count += 1
                        
                        # Save face image
                        face_img = gray[y:y+h, x:x+w]
                        filename = f"{name}.{student_id}.{count}.jpg"
                        cv2.imwrite(os.path.join(DIRS['training'], filename), face_img)
                        
                        # Draw on display frame
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
                        cv2.putText(display_frame, f"{count}/{max_count}", (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        if count >= max_count:
                            break
                
                # Update display with FULL frame
                self._show_frame(display_frame, self.reg_camera)
                
                # Update count label
                self.root.after(0, lambda c=count: self.capture_label.config(
                    text=f"Captured: {c}/{max_count}"))
            
            self.camera.release()
            self.camera = None
            self.running = False
            
            self.root.after(0, lambda: self._status(f"Done: {count} images"))
            self.root.after(0, lambda: self._clear_camera(self.reg_camera))
            
        except Exception as e:
            print(f"Capture error: {e}")
            self.root.after(0, lambda: self._status(f"Error: {e}"))
            if self.camera:
                self.camera.release()
            self.running = False
    
    def _show_frame(self, frame, label):
        """Display frame in label - FULL SIZE"""
        try:
            # Convert BGR to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get label size
            label.update_idletasks()
            lw = label.winfo_width()
            lh = label.winfo_height()
            
            if lw < 50 or lh < 50:
                lw, lh = 640, 480
            
            # Resize maintaining aspect ratio
            h, w = rgb.shape[:2]
            scale = min(lw/w, lh/h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            resized = cv2.resize(rgb, (new_w, new_h))
            
            # Convert to PhotoImage
            img = Image.fromarray(resized)
            photo = ImageTk.PhotoImage(img)
            
            # Update label from main thread
            def update():
                label.config(image=photo, text='')
                label.image = photo  # Keep reference
            
            self.root.after(0, update)
            
        except Exception as e:
            print(f"Display error: {e}")
    
    def _clear_camera(self, label):
        """Clear camera display"""
        label.config(image='', text='Camera')
        label.image = None
    
    def _start_training(self):
        """Start model training"""
        self.train_label.config(text="Training...")
        self._status("Training model...")
        threading.Thread(target=self._train_model, daemon=True).start()
    
    def _train_model(self):
        """Train recognition model"""
        try:
            images = [f for f in os.listdir(DIRS['training']) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not images:
                self.root.after(0, lambda: self.train_label.config(text="No images found"))
                self.root.after(0, lambda: self._status("No training data"))
                return
            
            faces = []
            labels = []
            
            for img_name in images:
                try:
                    parts = img_name.split('.')
                    if len(parts) >= 3:
                        label = int(parts[1])
                        img = cv2.imread(os.path.join(DIRS['training'], img_name), 
                                        cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            faces.append(img)
                            labels.append(label)
                except:
                    continue
            
            if not faces:
                self.root.after(0, lambda: self.train_label.config(text="No valid data"))
                return
            
            # Train
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.train(faces, np.array(labels))
            self.recognizer.save(os.path.join(DIRS['models'], "Trainner.yml"))
            
            self.root.after(0, lambda: self.train_label.config(
                text=f"Trained with {len(faces)} images"))
            self.root.after(0, lambda: self._status("Training complete"))
            
        except Exception as e:
            self.root.after(0, lambda: self.train_label.config(text=f"Error: {e}"))
            self.root.after(0, lambda: self._status("Training failed"))
    
    def _start_attendance(self):
        """Start attendance tracking"""
        subject = self.subject_entry.get().strip()
        if not subject:
            self._status("Enter subject")
            return
        
        self.running = True
        self._status("Tracking attendance...")
        threading.Thread(target=self._attendance_loop, args=(subject,), daemon=True).start()
    
    def _attendance_loop(self, subject):
        """Track attendance in background"""
        try:
            self.camera = cv2.VideoCapture(0)
            
            if not self.camera.isOpened():
                self.root.after(0, lambda: self._status("Camera not found"))
                return
            
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            marked = set()
            
            while self.running:
                ret, frame = self.camera.read()
                if not ret:
                    continue
                
                display = frame.copy()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if self.face_cascade is not None:
                    faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                    
                    for (x, y, w, h) in faces:
                        color = (255, 255, 255)
                        label_text = "Unknown"
                        
                        if self.recognizer:
                            try:
                                label, conf = self.recognizer.predict(gray[y:y+h, x:x+w])
                                
                                if conf < 70:
                                    if label not in marked:
                                        self._save_attendance(label, subject)
                                        marked.add(label)
                                    label_text = f"ID: {label}"
                                    color = (0, 255, 0)
                            except:
                                pass
                        
                        cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(display, label_text, (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                self._show_frame(display, self.att_camera)
                
                self.root.after(0, lambda m=len(marked): 
                    self.att_status.config(text=f"Marked: {m} students"))
            
            self.camera.release()
            self.camera = None
            self.root.after(0, lambda: self._clear_camera(self.att_camera))
            
        except Exception as e:
            print(f"Attendance error: {e}")
            if self.camera:
                self.camera.release()
    
    def _stop_attendance(self):
        """Stop attendance tracking"""
        self.running = False
        self._status("Stopped")
    
    def _save_attendance(self, student_id, subject):
        """Save attendance record"""
        try:
            now = datetime.now()
            record = {
                'ID': student_id,
                'Date': now.strftime("%Y-%m-%d"),
                'Time': now.strftime("%H:%M:%S"),
                'Subject': subject,
                'Status': 'Present'
            }
            
            filename = f"Attendance_{now.strftime('%Y-%m-%d')}.csv"
            filepath = os.path.join(DIRS['attendance'], filename)
            
            df = pd.DataFrame([record])
            if os.path.exists(filepath):
                df.to_csv(filepath, mode='a', header=False, index=False)
            else:
                df.to_csv(filepath, index=False)
        except Exception as e:
            print(f"Save error: {e}")
    
    def _open_reports(self):
        """Open reports folder"""
        try:
            if platform.system() == "Darwin":
                subprocess.Popen(['open', DIRS['attendance']])
            elif platform.system() == "Windows":
                subprocess.Popen(['explorer', DIRS['attendance']])
            else:
                subprocess.Popen(['xdg-open', DIRS['attendance']])
        except Exception as e:
            self._status(f"Error: {e}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("Starting Face Attendance System...")
    
    try:
        print(f"OpenCV: {cv2.__version__}")
    except:
        print("OpenCV not found")
        return
    
    try:
        cv2.face.LBPHFaceRecognizer_create()
        print("Face recognition: OK")
    except:
        print("Face recognition: Not available")
    
    root = tk.Tk()
    app = App(root)
    
    def on_close():
        app.running = False
        if app.camera:
            app.camera.release()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
