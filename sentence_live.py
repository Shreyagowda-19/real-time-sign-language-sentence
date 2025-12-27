import cv2
import numpy as np
import pickle
import time
from cvzone.HandTrackingModule import HandDetector

# Load trained model
with open("sign_model.pkl", "rb") as f:
    model = pickle.load(f)

label_map = {
    0: "Hello",
    1: "Yes",
    2: "No",
    3: "Thanks"
}

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1, detectionCon=0.8)

sentence = []
last_word = ""
last_time = 0
DELAY = 1.5  # seconds

current_prediction = ""

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
            pred = model.predict(X)[0]
            current_prediction = label_map[pred]

            current_time = time.time()

            if (
                current_prediction != last_word
                and current_time - last_time > DELAY
            ):
                sentence.append(current_prediction)
                last_word = current_prediction
                last_time = current_time

    # Display current word
    cv2.rectangle(img, (20, 20), (350, 90), (0, 0, 0), -1)
    cv2.putText(
        img,
        f"Word: {current_prediction}",
        (30, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    # Display sentence
    cv2.rectangle(img, (20, 110), (800, 200), (0, 0, 0), -1)
    cv2.putText(
        img,
        "Sentence: " + " ".join(sentence),
        (30, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    cv2.imshow("Sign Language Sentence Builder", img)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        sentence = []
        last_word = ""

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
