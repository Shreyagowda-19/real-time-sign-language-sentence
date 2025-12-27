import cv2
import os
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# Initialize webcam and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1, detectionCon=0.8)

# Settings
DATA_DIR = "data"
SAMPLES_PER_CLASS = 200

current_label = None
sample_count = 0

print("Press keys to start collecting data:")
print("h → Hello | y → Yes | n → No | t → Thanks | q → Quit")

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img, draw=True)

    cv2.putText(img, f"Label: {current_label}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, f"Samples: {sample_count}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    if hands and current_label is not None:
        hand = hands[0]
        landmarks = []

        for lm in hand["lmList"]:
            landmarks.extend(lm)  # x, y, z

        if len(landmarks) == 63:
            file_path = os.path.join(
                DATA_DIR,
                current_label,
                f"{sample_count}.npy"
            )
            np.save(file_path, np.array(landmarks))
            sample_count += 1

    cv2.imshow("Data Collection", img)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('h'):
        current_label = "Hello"
        sample_count = 0
        print("Collecting HELLO")

    elif key == ord('y'):
        current_label = "Yes"
        sample_count = 0
        print("Collecting YES")

    elif key == ord('n'):
        current_label = "No"
        sample_count = 0
        print("Collecting NO")

    elif key == ord('t'):
        current_label = "Thanks"
        sample_count = 0
        print("Collecting THANKS")

    elif key == ord('q'):
        break

    if sample_count >= SAMPLES_PER_CLASS:
        print(f"Finished collecting {current_label}")
        current_label = None
        sample_count = 0

cap.release()
cv2.destroyAllWindows()
