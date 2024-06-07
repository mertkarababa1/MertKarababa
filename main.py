import cv2
import numpy as np
from keras.models import load_model
import librosa
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Load pre-trained models for image and audio emotion recognition
image_model = load_model('/Users/mertkarababa/Desktop/mk8/.venv/models/cam_model.h5')
audio_model = load_model('/Users/mertkarababa/Desktop/mk8/.venv/models/best_complex_model1.keras')

# Load Haar cascade for face detection
face_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Emotion labels for the predictions
emotion_labels = ["Angry", "Disgusted", "Afraid", "Happy", "Neutral", "Sad", "Surprised"]

# Initialize the camera for video capture
camera = cv2.VideoCapture(0)

# Global variables to store emotion results
emotion_result = None
audio_emotion_result = None

def process_frame():
    # Capture frame-by-frame
    success, frame = camera.read()
    if not success:
        return None

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, 1.3, 5)
    
    # Process each face detected in the frame
    for (x, y, w, h) in faces:
        sub_face_img = gray[y:y + h, x:x + w]
        resized = cv2.resize(sub_face_img, (48, 48))
        normalize = resized / 255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        result = image_model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]

        global emotion_result
        emotion_result = {
            'label': emotion_labels[label],
            'score': float(np.max(result)) * 10  # Convert the score to a 0-10 range
        }
        break  # Process only one face and exit

    return frame

def process_audio(file_path):
    try:
        # Load the audio file
        audio, sr = librosa.load(file_path, sr=None, res_type='kaiser_fast')
        # Extract MFCC features from the audio
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfccs = np.mean(mfccs.T, axis=0)
        mfccs = np.expand_dims(mfccs, axis=0)
        mfccs = np.expand_dims(mfccs, axis=2)  # Added to make it suitable for the GRU model
        # Predict emotion from audio features
        predictions = audio_model.predict(mfccs)
        score = np.max(predictions) * 10
        emotion_index = np.argmax(predictions)
        emotion = emotion_labels[emotion_index]
        return {'score': score, 'emotion': emotion}
    except Exception as e:
        print(f"Error encountered while processing audio: {file_path}, Error: {e}")
        return {'score': 0, 'emotion': 'Error'}

def update_frame():
    # Update the frame in the Tkinter window
    frame = process_frame()
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
    lmain.after(10, update_frame)

def upload_audio():
    # Function to upload and process audio file
    file_path = filedialog.askopenfilename()
    if file_path:
        global audio_emotion_result
        audio_emotion_result = process_audio(file_path)
        messagebox.showinfo("Audio Emotion Result", f"Emotion: {audio_emotion_result['emotion']}, Score: {audio_emotion_result['score']}")

# Initialize the Tkinter root window
root = tk.Tk()
root.title("Emotion Detection")
root.configure(bg='#800000')  # Set background color to dark-red

# Add university label at the top
university_label = tk.Label(root, text="T.C. Beykoz Üniversitesi Yazılım Mühendisliği", font=("Arial", 24), bg='#800000', fg='white')
university_label.pack(pady=10)

# Add title label below the university label
title_label = tk.Label(root, text="Mert Karababa", font=("Arial", 18), bg='#800000', fg='white')
title_label.pack(pady=10)

# Add label to display video frames
lmain = tk.Label(root)
lmain.pack()

# Add button to upload audio files
btn_upload_audio = tk.Button(root, text="Upload Audio", command=upload_audio, bg='#800000', fg='white', font=("Arial", 14))
btn_upload_audio.pack(pady=10)

# Add label to display emotion results
emotion_label = tk.Label(root, text="", font=("Arial", 18), bg='#800000', fg='white')
emotion_label.pack(pady=10)

def update_emotion_label():
    # Update the emotion label in the Tkinter window
    global emotion_result
    if emotion_result:
        emotion_label.config(text=f"Emotion: {emotion_result['label']}, Score: {emotion_result['score']}")
    root.after(1000, update_emotion_label)

# Start updating frames and emotion labels
update_frame()
update_emotion_label()

# Run the Tkinter main loop
root.mainloop()
