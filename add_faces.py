import cv2
import pickle
import numpy as np
import os

# Create directory if not exists
if not os.path.exists('face_data'):
    os.makedirs('face_data')

# Initialize webcam
video = cv2.VideoCapture(0)

# Load the face detection model
facesdetect = cv2.CascadeClassifier("face_data/cascade.xml")

# Check webcam and classifier
if not video.isOpened():
    print("Error: Could not open webcam.")
    exit()

if facesdetect.empty():
    print("Error: Could not load cascade classifier.")
    exit()

# Ask once for user name
name = input("Enter your name: ")
faces_data = []
count = 0

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facesdetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resize_img = cv2.resize(crop_img, (50, 50))
        
        if count % 10 == 0 and len(faces_data) < 100:
            faces_data.append(resize_img)
        
        count += 1

        cv2.putText(frame, f"Collected: {len(faces_data)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow("Face Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or len(faces_data) >= 100:
        break

video.release()
cv2.destroyAllWindows()

# Convert to numpy array and reshape
faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(100, -1)

# Load or create names and faces_data
names_file = 'face_data/names.pkl'
faces_file = 'face_data/faces_data.pkl'

if os.path.exists(names_file):
    with open(names_file, 'rb') as f:
        names = pickle.load(f)
else:
    names = []

if os.path.exists(faces_file):
    with open(faces_file, 'rb') as f:
        faces = pickle.load(f)
else:
    faces = np.empty((0, 7500))  # 50x50x3 = 7500

# Append new data
names.extend([name] * 100)
faces = np.append(faces, faces_data, axis=0)

# Save updated data
with open(names_file, 'wb') as f:
    pickle.dump(names, f)

with open(faces_file, 'wb') as f:
    pickle.dump(faces, f)

print(f"Saved data for: {name}")
