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
    image_path = path + _
    encoding = fr.face_encodings(image)
    if encoding:  # Only add the encoding if a face is detected
        known_name_encodings.append(encoding[0])
        known_names.append(os.path.splitext(os.path.basename(image_path))[0].capitalize())

print("Known Names:", known_names)

# Load Age and Gender models
gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')

GENDER_LIST = ['Male', 'Female']

# Path to the test image
test_image = "./test/img7.jpg"
image = cv2.imread(test_image)

# Detect faces in the uploaded image
face_locations = fr.face_locations(image)
face_encodings = fr.face_encodings(image, face_locations)

# Load or create an Excel file
excel_file = "face_log.xlsx"
if os.path.exists(excel_file):
    df = pd.read_excel(excel_file)
else:
    df = pd.DataFrame(columns=["Name", "Time", "Access", "Emotion", "Gender"])

# Initialize emotion detector
emotion_detector = FER(mtcnn=True)

# Check if any faces were detected
if not face_locations:
    print("No faces detected in the image.")
else:
    # Process each detected face
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = fr.compare_faces(known_name_encodings, face_encoding)
        name = "Unknown"  # Default name for unidentified faces
        access = "Unauthorized"  # Default access status for unidentified faces

        # Check for the best match
        face_distances = fr.face_distance(known_name_encodings, face_encoding)
        best_match = np.argmin(face_distances)

        if matches[best_match]:
            name = known_names[best_match]
            access = "Authorized"
            print(f"Recognized: {name}")

        # Extract the face region for emotion detection
        face_image = image[top:bottom, left:right]
        face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        emotion_result = emotion_detector.detect_emotions(face_image_rgb)

        # Extract dominant emotion
        emotion = "Not Detected"
        if emotion_result:
            emotion = max(emotion_result[0]["emotions"], key=emotion_result[0]["emotions"].get)
            print(f"Emotion detected for {name}: {emotion}")

        # Age and Gender Prediction
        blob = cv2.dnn.blobFromImage(face_image, 1, (227, 227), (78.5, 87.5, 115), swapRB=False, crop=False)
        
        # Predict Gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = GENDER_LIST[gender_preds[0].argmax()]


        print("Gender: {gender}")

        # Log the result in the Excel file
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_entry = {"Name": name, "Time": now, "Access": access, "Emotion": emotion, "Gender": gender}
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)

        # Draw rectangle around face and annotate with name, emotion, age, and gender
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(image, (left, bottom - 15), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, f"Name: {name}", (left + 6, bottom - 3), font, 0.6, (255, 255, 255), 1)
        cv2.putText(image, f"Mood: {emotion}", (left + 6, bottom + 14), font, 0.6, (255, 255, 255), 1)
        cv2.putText(image, f"Gender: {gender}", (left + 6, bottom + 31), font, 0.6, (255, 255, 255), 1)

    # Save updated log to the Excel file
    df.to_excel(excel_file, index=False)
    print("Log updated in face_log_file.xlsx")

    # Show the result
    cv2.imshow("Result", image)
    cv2.imwrite("./output.jpg", image)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()



# print(f"Loading image: {image_path}")
# print(f"Encoding for {os.path.splitext(os.path.basename(image_path))[0]} added.")
# print(f"Processing test image: {test_image}")
# print(f"Detected face locations: {face_locations}")
# print(f"Face distances: {face_distances}")
# print(f"Best match index: {best_match}")
# print(f"Is the match valid? {matches[best_match]}")
# print(f"Detected emotions: {emotion_result}")
# print(f"Dominant emotion: {emotion}")
# print(f"Gender prediction scores: {gender_preds}")
# print(f"Predicted gender: {gender}")
# print(f"New entry logged: {new_entry}")

