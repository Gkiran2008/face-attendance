from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, "data")
CASCADE_PATH = os.path.join(DATA_PATH, "haarcascade_frontalface_default.xml")
NAMES_FILE = os.path.join(DATA_PATH, "names.pkl")
FACES_FILE = os.path.join(DATA_PATH, "faces_data.pkl")

# Load cascade
facedetect = cv2.CascadeClassifier(CASCADE_PATH)

# Check files
if not os.path.exists(NAMES_FILE) or not os.path.exists(FACES_FILE):
    print("❌ No training data found! Please register faces first using add_faces.py.")
    exit()

# Load training data
with open(NAMES_FILE, 'rb') as f:
    LABELS = pickle.load(f)
with open(FACES_FILE, 'rb') as f:
    FACES = pickle.load(f)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Start camera
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)

        # Draw UI
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x+5, y-10),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

    cv2.imshow("Secure Attendance System - Recognition", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
