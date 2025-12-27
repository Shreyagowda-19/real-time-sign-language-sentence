# Real-Time Sign Language Sentence Detection

This project detects hand gestures using a webcam and converts them into words and sentences in real time using machine learning.

## Features
- Real-time hand tracking using CVZone (MediaPipe)
- Word detection: Hello, Yes, No, Thanks
- Sentence formation with time-based filtering
- Random Forest classifier for gesture recognition
- Live webcam-based prediction

## Technologies Used
- Python 3.10
- OpenCV
- CVZone
- MediaPipe
- NumPy
- Scikit-learn

## How It Works
1. Hand landmarks (21 points) are extracted from webcam feed
2. Landmarks are used as numerical features
3. ML model predicts the gesture
4. Predictions are combined into sentences in real time

## How to Run
```bash
pip install opencv-python mediapipe cvzone numpy scikit-learn
python sentence_live.py
