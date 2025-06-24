🔍 Models and Algorithms Used

1. Haar Cascade Classifier (Face Detection)**

* **Library**: OpenCV
* **Model**: `cascade.xml`
* **Purpose**: Detects human faces in webcam video frames.
* **How it works**: Haar features are simple rectangular features computed quickly using integral images. The classifier uses AdaBoost and a cascade structure to improve detection speed and accuracy.

---

### 2. **K-Nearest Neighbors (KNN) Classifier (Face Recognition)**

* **Library**: scikit-learn (`KNeighborsClassifier`)
* **Purpose**: Identifies the person by comparing the detected face with stored face data.
* **How it works**: Given a new face image, KNN finds the ‘k’ closest matches from the training set using Euclidean distance and predicts the most common label (name).

---

### 3. **Pickle Serialization (Model & Data Storage)**

* **Library**: Python `pickle`
* **Purpose**: Saves face data (`faces_data.pkl`) and corresponding labels (`names.pkl`) to disk.
* **How it works**: Converts Python objects into byte streams to persist data between sessions (used during training and recognition).

---

### 4. **Text-to-Speech (Voice Feedback)**

* **Library**: `pywin32` via `win32com.client.Dispatch("SAPI.SpVoice")`
* **Purpose**: Provides audio feedback when attendance is marked.
* **How it works**: Windows SAPI (Speech API) speaks out the name of the person detected.

---

### 5. **NumPy (Image Processing)**

* **Library**: `numpy`
* **Purpose**: Stores, reshapes, and manipulates image data (face arrays).
* **How it works**: Image arrays are flattened and reshaped into vectors for model training and prediction.

---

### 6. **CSV Handling (Attendance Logs)**

* **Library**: Python `csv`
* **Purpose**: Saves attendance records in a date-wise `.csv` format.
* **How it works**: Writes rows like `[name, timestamp]` into a file only once per person per session.

---

## 🛠️ Summary Flow

1. Face is detected using Haar Cascade.
2. The detected face is resized and flattened.
3. KNN predicts the person's name based on stored data.
4. Attendance is recorded in a `.csv` file with a timestamp.
5. A voice announces the person's name if attendance is marked.

---
🚀 How to Run
git clone https://github.com/shristi76/smart-attendance-system.git
cd face-recognition-attendance

then run add_faces.py  for capturing the face image
then run test.py
