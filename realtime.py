import face_recognition as fr
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
from fer import FER  # Install using: pip install fer

# Path to training images
path = "./train/"

# Initialize known names and encodings
known_names = []
known_name_encodings = []

# Load and encode faces from training images
images = os.listdir(path)
for _ in images:
    image = fr.load_image_file(path + _)
    encoding = fr.face_encodings(image)
    if encoding:  # Only add the encoding if a face is detected
        known_name_encodings.append(encoding[0])
        known_names.append(os.path.splitext(os.path.basename(_))[0].capitalize())

print("Known Names:", known_names)

# Load Gender Prediction Model
gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')
GENDER_LIST = ['Male', 'Female']

# Load or create an Excel file
excel_file = "face_log_test.xlsx"
if os.path.exists(excel_file):
    df = pd.read_excel(excel_file)
else:
    df = pd.DataFrame(columns=["Name", "Time", "Access", "Emotion", "Gender"])

logged_names = set(df["Name"])  # Track already logged names

# Initialize emotion detector
emotion_detector = FER(mtcnn=True)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting.")
        break

    # Detect faces in the frame
    face_locations = fr.face_locations(frame)
    face_encodings = fr.face_encodings(frame, face_locations)

    # Process each detected face
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = fr.compare_faces(known_name_encodings, face_encoding)
        name = "Unknown"
        access = "Unauthorized"

        # Check for the best match
        face_distances = fr.face_distance(known_name_encodings, face_encoding)
        best_match = np.argmin(face_distances)

        if matches[best_match]:
            name = known_names[best_match]
            access = "Authorized"

        # Extract the face region for emotion detection
        face_image = frame[top:bottom, left:right]
        face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        emotion_result = emotion_detector.detect_emotions(face_image_rgb)

        # Extract dominant emotion
        emotion = "Not Detected"
        if emotion_result:
            emotion = max(emotion_result[0]["emotions"], key=emotion_result[0]["emotions"].get)

        # Predict Gender
        blob = cv2.dnn.blobFromImage(face_image, 1, (227, 227), (78.5, 87.5, 115), swapRB=False, crop=False)
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = GENDER_LIST[gender_preds[0].argmax()]

        # Log the result only if not already logged
        if name not in logged_names:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_entry = {"Name": name, "Time": now, "Access": access, "Emotion": emotion, "Gender": gender}
            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
            logged_names.add(name)

        # Draw rectangle around face and annotate with details
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, f"Name: {name}", (left + 6, top - 10), font, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Access: {access}", (left + 6, top + 20), font, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Mood: {emotion}", (left + 6, top + 40), font, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Gender: {gender}", (left + 6, top + 60), font, 0.6, (255, 255, 255), 1)

    # Display the frame
    cv2.imshow("Real-Time Face Recognition", frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save updated log to the Excel file
df.to_excel(excel_file, index=False)
print(f"Log updated in {excel_file}")

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
