from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

# Text-to-speech function
def speak(text):
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(text)

# Load face detection model
facesdetect = cv2.CascadeClassifier("face_data/cascade.xml")
if facesdetect.empty():
    print("Error: Could not load cascade classifier.")
    exit()

# Load face data and labels
try:
    with open('face_data/names.pkl', 'rb') as f:
        LABEL = pickle.load(f)
    with open('face_data/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)
except FileNotFoundError:
    print("Error: Data files not found.")
    exit()

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABEL)

# Open webcam
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error: Webcam not accessible.")
    exit()

# Optional background image
img_bg = cv2.imread('background.png')
if img_bg is None:
    img_bg = np.ones((720, 1280, 3), dtype=np.uint8) * 255  # fallback white

col_name = ["NAME", "TIME"]
recorded_names = set()

while True:
    ret, frame = video.read()
    if not ret:
        print("Frame grab failed.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facesdetect.detectMultiScale(gray, 1.3, 5)

    attendance = None

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resize_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resize_img)
        name = output[0]

        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        attendance = [name, timestamp]

        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    img_bg[162:162 + 480, 55:55 + 640] = frame
    cv2.imshow("Face Recognition Attendance", img_bg)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('o') and attendance is not None:
        name = attendance[0]
        file_path = f"attendance/attendance_{date}.csv"
        os.makedirs("attendance", exist_ok=True)

        if name not in recorded_names:
            recorded_names.add(name)
            speak(f"Welcome {name}, your attendance has been marked.")
            write_header = not os.path.exists(file_path)
            with open(file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(col_name)
                writer.writerow(attendance)
            print(f"Attendance marked for {name} at {attendance[1]}")

video.release()
cv2.destroyAllWindows()
