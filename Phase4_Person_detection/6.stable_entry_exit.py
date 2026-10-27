# ============================================================
# STABLE PERSON ENTRY / EXIT DETECTION
# ============================================================

import cv2
from ultralytics import YOLO


# ------------------------------------------------------------
# 1. LOAD YOLO MODEL
# ------------------------------------------------------------

model = YOLO("yolo11n.pt")


# ------------------------------------------------------------
# 2. START WEBCAM
# ------------------------------------------------------------

cap = cv2.VideoCapture(0)


# ------------------------------------------------------------
# 3. TRACKING SETTINGS
# ------------------------------------------------------------

# Minimum confidence for detecting a person
CONFIDENCE = 0.5

# Number of frames a person must be missing
# before we consider them to have left
MAX_MISSING_FRAMES = 120


# ------------------------------------------------------------
# 4. STORE TRACKING INFORMATION
# ------------------------------------------------------------

# IDs that were confirmed as present
confirmed_ids = set()

# Number of consecutive frames each ID is missing
missing_frames = {}


# ------------------------------------------------------------
# 5. MAIN LOOP
# ------------------------------------------------------------

while True:

    # Read a frame from the webcam
    ret, frame = cap.read()

    # Stop if camera frame cannot be read
    if not ret:
        break


    # --------------------------------------------------------
    # 6. RUN YOLO TRACKING
    # --------------------------------------------------------

    results = model.track(
        frame,
        conf=CONFIDENCE,
        persist=True,
        verbose=False
    )

    # Get the first result
    result = results[0]


    # IDs detected in the current frame
    current_ids = set()


    # --------------------------------------------------------
    # 7. GET PERSON TRACKING IDs
    # --------------------------------------------------------

    if result.boxes.id is not None:

        # Get tracking IDs
        track_ids = result.boxes.id.int().cpu().tolist()

        # Get object classes
        class_ids = result.boxes.cls.int().cpu().tolist()


        # Check every detected object
        for track_id, class_id in zip(track_ids, class_ids):

            # Get object name
            class_name = model.names[class_id]


            # Only process people
            if class_name == "person":

                # Add the ID to current detections
                current_ids.add(track_id)


    # --------------------------------------------------------
    # 8. DETECT NEW STABLE PERSON
    # --------------------------------------------------------

    for person_id in current_ids:

        # If this ID has never been confirmed
        if person_id not in confirmed_ids:

            # Consider the person entered
            print(f"PERSON ENTERED | ID: {person_id}")

            # Add ID to confirmed people
            confirmed_ids.add(person_id)

        # Person is currently visible
        missing_frames[person_id] = 0


    # --------------------------------------------------------
    # 9. CHECK FOR MISSING PEOPLE
    # --------------------------------------------------------

    for person_id in list(confirmed_ids):

        # If the person is not visible
        if person_id not in current_ids:

            # Increase missing-frame counter
            missing_frames[person_id] = (
                missing_frames.get(person_id, 0) + 1
            )


            # ------------------------------------------------
            # 10. CONFIRM EXIT
            # ------------------------------------------------

            if missing_frames[person_id] >= MAX_MISSING_FRAMES:

                # Person has actually left
                print(f"PERSON LEFT | ID: {person_id}")

                # Remove the ID
                confirmed_ids.remove(person_id)

                # Remove its missing counter
                del missing_frames[person_id]


    # --------------------------------------------------------
    # 11. DISPLAY RESULT
    # --------------------------------------------------------

    annotated_frame = result.plot()

    cv2.imshow(
        "Stable Entry Exit Detection",
        annotated_frame
    )


    # --------------------------------------------------------
    # 12. PRESS Q TO EXIT
    # --------------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ------------------------------------------------------------
# 13. RELEASE CAMERA
# ------------------------------------------------------------

cap.release()

# Close OpenCV windows
cv2.destroyAllWindows()