
import cv2
import os
import time
from ultralytics import YOLO

# ==================================================
# SETTINGS
# ==================================================

CONFIDENCE = 0.50

# Weapon model file
MODEL_PATH = "weapon_model.pt"

# ==================================================
# CHECK MODEL
# ==================================================

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: {MODEL_PATH} not found.")
    print("Place your weapon detection .pt model in this folder.")
    exit()

# Load weapon detection model
model = YOLO(MODEL_PATH)

# ==================================================
# OPEN WEBCAM
# ==================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("WEAPON DETECTION STARTED")
print("Press Q to quit.")

# Used to avoid printing the same detection repeatedly
last_detection_time = 0
DETECTION_COOLDOWN = 2.0

# ==================================================
# MAIN LOOP
# ==================================================

while True:

    # Read webcam frame
    success, frame = cap.read()

    if not success:
        print("ERROR: Could not read webcam frame.")
        break

    # Run weapon detection
    results = model(
        frame,
        conf=CONFIDENCE,
        verbose=False
    )

    # Get the first result
    result = results[0]

    weapon_detected = False

    # ==================================================
    # PROCESS DETECTIONS
    # ==================================================

    if result.boxes is not None:

        for box in result.boxes:

            # Get confidence
            confidence = float(box.conf[0])

            # Get class ID
            class_id = int(box.cls[0])

            # Get class name
            class_name = model.names[class_id]

            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            weapon_detected = True

            # Draw red bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

            # Detection label
            label = f"{class_name} | {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

    # ==================================================
    # WEAPON ALERT
    # ==================================================

    if weapon_detected:

        # Display alert on screen
        cv2.putText(
            frame,
            "WEAPON DETECTED!",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

        # Print event only once every few seconds
        current_time = time.time()

        if current_time - last_detection_time >= DETECTION_COOLDOWN:

            print("WEAPON DETECTED")

            last_detection_time = current_time

    else:

        # Normal status
        cv2.putText(
            frame,
            "STATUS: CLEAR",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    # ==================================================
    # DISPLAY FRAME
    # ==================================================

    cv2.imshow("AI Weapon Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==================================================
# CLEANUP
# ==================================================

cap.release()
cv2.destroyAllWindows()

print("WEAPON DETECTION STOPPED")

