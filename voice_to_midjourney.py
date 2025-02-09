import sys
import json
import queue
import sounddevice as sd
import numpy as np
import whisper
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                            QWidget, QLabel, QMessageBox, QStackedWidget, QFileDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve, QTimer, QSize
from PyQt6.QtGui import QColor, QPainter, QBrush, QPen, QFont, QFontDatabase
import openai
from PIL import Image
from io import BytesIO

class LoadingAnimation(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.angle = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.rotate)
        self.timer.start(50)  # Update every 50ms
        
    def rotate(self):
        self.angle = (self.angle + 10) % 360
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Create loading circle
        pen = QPen(QColor(255, 255, 255))
        pen.setWidth(4)
        painter.setPen(pen)
        
        center = self.rect().center()
        radius = min(self.width(), self.height()) // 4
        
        painter.translate(center)
        painter.rotate(self.angle)
        
        for i in range(8):
            painter.rotate(45)
            opacity = (i + 1) / 8.0
            pen.setColor(QColor(255, 255, 255, int(255 * opacity)))
            painter.setPen(pen)
            painter.drawLine(radius - 10, 0, radius, 0)

class AudioRecorder(QThread):
    finished = pyqtSignal(np.ndarray)
    
    def __init__(self, sample_rate=16000):
        super().__init__()
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
    def run(self):
        def callback(indata, frames, time, status):
            if status:
                print(status)
            self.audio_queue.put(indata.copy())
            
        with sd.InputStream(callback=callback, channels=1, 
                          samplerate=self.sample_rate):
            while self.is_recording:
                sd.sleep(100)
        
        audio_data = []
        while not self.audio_queue.empty():
            audio_data.append(self.audio_queue.get())
        
        if audio_data:
            combined = np.concatenate(audio_data)
            self.finished.emit(combined)
    
    def start_recording(self):
        self.is_recording = True
        self.start()
    
    def stop_recording(self):
        self.is_recording = False

class ProcessingThread(QThread):
    finished = pyqtSignal(str, str)
    error = pyqtSignal(str)
    
    def __init__(self, audio_data, whisper_model, openai_client):
        super().__init__()
        self.audio_data = audio_data
        self.whisper_model = whisper_model
        self.openai_client = openai_client
    
    def run(self):
        try:
            transcription = self.whisper_model.transcribe(self.audio_data)
            spoken_text = transcription["text"]
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that converts spoken descriptions into detailed MidJourney prompts. Focus on visual details and artistic style."},
                    {"role": "user", "content": f"Convert this description into a detailed MidJourney prompt: {spoken_text}"}
                ]
            )
            
            image_prompt = response.choices[0].message.content
            self.finished.emit(spoken_text, image_prompt)
            
        except Exception as e:
            self.error.emit(str(e))

class ArtDisplayWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: black;")
        self.image = None
        self.fade_animation = QPropertyAnimation(self, b"windowOpacity")
        self.fade_animation.setDuration(1000)
        self.fade_animation.setEasingCurve(QEasingCurve.Type.InOutCubic)
        
    def set_image(self, image_data):
        self.image = Image.open(BytesIO(image_data))
        self.update()
        
    def paintEvent(self, event):
        if self.image:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            # Scale image to fit window while maintaining aspect ratio
            scaled_size = QSize(self.width(), self.height())
            img_ratio = self.image.width / self.image.height
            window_ratio = self.width() / self.height()
            
            if window_ratio > img_ratio:
                scaled_size.setWidth(int(self.height() * img_ratio))
            else:
                scaled_size.setHeight(int(self.width() / img_ratio))
            
            # Convert PIL Image to QImage and draw
            img_qt = self.image.toqpilimage()
            img_qt = img_qt.scaled(scaled_size, Qt.AspectRatioMode.KeepAspectRatio, 
                                 Qt.TransformationMode.SmoothTransformation)
            
            # Center the image
            x = (self.width() - scaled_size.width()) // 2
            y = (self.height() - scaled_size.height()) // 2
            painter.drawImage(x, y, img_qt)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.load_config()
        self.setup_models()
        
    def init_ui(self):
        self.setWindowTitle('Voice to Art')
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.showFullScreen()
        
        # Create stacked widget for different views
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        # Create recording view
        self.recording_widget = QWidget()
        recording_layout = QVBoxLayout(self.recording_widget)
        
        # Style the recording interface
        self.record_button = QPushButton('Press Space to Start Recording')
        self.record_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 0.7);
                color: white;
                border: 2px solid white;
                border-radius: 20px;
                padding: 20px;
                font-size: 24px;
            }
            QPushButton:hover {
                background-color: rgba(40, 40, 40, 0.7);
            }
        """)
        self.record_button.clicked.connect(self.toggle_recording)
        
        # Add loading animation (hidden initially)
        self.loading_animation = LoadingAnimation()
        self.loading_animation.hide()
        
        recording_layout.addStretch()
        recording_layout.addWidget(self.record_button, alignment=Qt.AlignmentFlag.AlignCenter)
        recording_layout.addWidget(self.loading_animation, alignment=Qt.AlignmentFlag.AlignCenter)
        recording_layout.addStretch()
        
        # Create art display view
        self.art_display = ArtDisplayWidget()
        
        # Add widgets to stacked widget
        self.stacked_widget.addWidget(self.recording_widget)
        self.stacked_widget.addWidget(self.art_display)
        
        # Initialize state
        self.is_recording = False
        self.recorder = None
        
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.toggle_recording()
        elif event.key() == Qt.Key.Key_Escape:
            self.close()
            
    def load_config(self):
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
                self.openai_api_key = config['openai_api_key']
        except FileNotFoundError:
            QMessageBox.critical(self, 'Error', 'config.json file not found!')
            sys.exit(1)
            
    def setup_models(self):
        self.whisper_model = whisper.load_model("small")
        self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
            
    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
            
    def start_recording(self):
        self.record_button.setText('Press Space to Stop Recording')
        self.is_recording = True
        self.recorder = AudioRecorder()
        self.recorder.finished.connect(self.process_audio)
        self.recorder.start_recording()
        
    def stop_recording(self):
        self.record_button.setText('Processing...')
        self.record_button.hide()
        self.loading_animation.show()
        self.is_recording = False
        if self.recorder:
            self.recorder.stop_recording()
            
    def process_audio(self, audio_data):
        self.processing_thread = ProcessingThread(
            audio_data, self.whisper_model, self.openai_client)
        self.processing_thread.finished.connect(self.submit_to_midjourney)
        self.processing_thread.error.connect(self.show_error)
        self.processing_thread.start()
        
    def show_error(self, error_message):
        self.loading_animation.hide()
        self.record_button.show()
        self.record_button.setText('Press Space to Start Recording')
        QMessageBox.critical(self, 'Error', error_message)
        
    def submit_to_midjourney(self, transcription, prompt):
        try:
            # Copy the /imagine command to clipboard
            clipboard = QApplication.clipboard()
            clipboard.setText(f"/imagine {prompt}")
            
            # Show instruction dialog
            msg = QMessageBox()
            msg.setStyleSheet("""
                QMessageBox {
                    background-color: rgba(0, 0, 0, 0.9);
                    color: white;
                }
                QPushButton {
                    background-color: rgba(40, 40, 40, 0.9);
                    color: white;
                    border: 1px solid white;
                    padding: 5px 15px;
                    border-radius: 3px;
                }
                QLabel {
                    color: white;
                }
            """)
            msg.setWindowTitle("MidJourney Prompt")
            msg.setText("The MidJourney command has been copied to your clipboard.\n\n" + 
                       "1. Open Discord and go to your MidJourney DMs\n" +
                       "2. Paste the command (Ctrl+V or Cmd+V)\n" +
                       "3. Once the image is generated, right-click and save it\n" +
                       "4. Click 'Select Image' below to load it into the app")
            msg.addButton("Select Image", QMessageBox.ButtonRole.AcceptRole)
            msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            
            response = msg.exec()
            
            if response == QMessageBox.ButtonRole.AcceptRole:
                # Open file dialog to select the saved image
                file_name, _ = QFileDialog.getOpenFileName(
                    self,
                    "Select Generated Image",
                    "",
                    "Image Files (*.png *.jpg *.jpeg)"
                )
                
                if file_name:
                    with open(file_name, 'rb') as f:
                        image_data = f.read()
                    self.art_display.set_image(image_data)
                    self.stacked_widget.setCurrentWidget(self.art_display)
                    
                    # Reset recording interface
                    self.loading_animation.hide()
                    self.record_button.show()
                    self.record_button.setText('Press Space to Start Recording')
                else:
                    self.show_error("No image selected")
            else:
                self.show_error("Operation cancelled")
                
        except Exception as e:
            self.show_error(f"Error: {str(e)}")
                
        except Exception as e:
            self.show_error(f'Failed to submit: {str(e)}')
            
    def poll_for_image(self):
        # In a real implementation, you would need to poll Discord/MidJourney API
        # to get the actual generated image. For now, we'll simulate with a delay
        QTimer.singleShot(5000, self.transition_to_art)
        
    def transition_to_art(self):
        # Here you would fetch the actual generated image URL from MidJourney
        # For now, we'll use a placeholder image
        try:
            # Replace this with actual MidJourney image URL when available
            response = requests.get("https://picsum.photos/1920/1080")
            if response.status_code == 200:
                self.art_display.set_image(response.content)
                self.stacked_widget.setCurrentWidget(self.art_display)
                
                # Reset recording interface for next use
                self.loading_animation.hide()
                self.record_button.show()
                self.record_button.setText('Press Space to Start Recording')
            else:
                self.show_error("Failed to load generated image")
        except Exception as e:
            self.show_error(f"Failed to load image: {str(e)}")

def main():
    app = QApplication(sys.argv)
    
    # Set application-wide stylesheet
    app.setStyleSheet("""
        QMainWindow {
            background-color: black;
        }
    """)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()