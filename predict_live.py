import cv2
import numpy as np
import pickle
from cvzone.HandTrackingModule import HandDetector

# Load trained model
with open("sign_model.pkl", "rb") as f:
    model = pickle.load(f)

# Label mapping
label_map = {
    0: "Hello",
    1: "Yes",
    2: "No",
    3: "Thanks"
}

# Initialize webcam and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1, detectionCon=0.8)

predicted_word = ""

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img, draw=True)

    if hands:
        hand = hands[0]
        landmarks = []

        for lm in hand["lmList"]:
            landmarks.extend(lm)

        if len(landmarks) == 63:
            X = np.array(landmarks).reshape(1, -1)
            prediction = model.predict(X)[0]
            predicted_word = label_map[prediction]

    # Display predicted word
    cv2.rectangle(img, (20, 20), (350, 100), (0, 0, 0), -1)
    cv2.putText(
        img,
        f"Prediction: {predicted_word}",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 0),
        3
    )

    cv2.imshow("Live Sign Prediction", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
