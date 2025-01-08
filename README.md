# Face Security System

This project implements a face security system that performs **face recognition**, **emotion detection**, and **gender classification** from images or live webcam footage. The system identifies faces, detects emotions (e.g., happiness, sadness), and classifies gender, logging the results for future reference.

## Key Features
- **Face Recognition**: Identifies individuals based on pre-trained encodings.
- **Emotion Detection**: Detects emotions (happy, sad, neutral) using the FER library.
- **Gender Classification**: Classifies gender (male or female) using a pre-trained Caffe model.
- **Results Logging**: Logs recognized names, emotions, gender, and access status in an Excel file.

## Dependencies
- `face_recognition`
- `opencv-python`
- `FER`
- `pandas`
- `datetime`

## Installation
1. Clone the repo:
2. Install dependencies:
3. Run the system:
   ```bash
   python main.py
   ```
   or
   
    ```bash
   python realtime.py
   ```
   

## Usage
- **Train images**: Store images in the `./train/` folder for face recognition.
- **Test images**: Place test images in the `./test/` folder for face detection, emotion, and gender prediction.

The system will log the results in `face_log.xlsx`.

---

This concise README covers the essentials of the project!
