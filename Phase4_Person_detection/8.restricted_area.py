
import cv2
import os
import time
from datetime import datetime
from ultralytics import YOLO

# ==================================================
# SETTINGS
# ==================================================

CONFIDENCE = 0.5

# Frames a person can be missing before being considered LEFT
MAX_MISSING_FRAMES = 100

# Person must stay inside restricted area for this many seconds
SNAPSHOT_DELAY = 2.0

# Folder where security snapshots will be saved
SNAPSHOT_FOLDER = "snapshots"

# Create snapshot folder if it does not exist
os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)

# ==================================================
# LOAD YOLO MODEL
# ==================================================

model = YOLO("yolo11n.pt")

# ==================================================
# OPEN WEBCAM
# ==================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

# ==================================================
# PERSON ID MANAGEMENT
# ==================================================

next_person_id = 1

# YOLO tracker ID -> our person ID
yolo_to_person = {}

# Currently active people
active_people = {}

# Missing people and their missing frame count
missing_people = {}

# IDs that already triggered the restricted-area event
alerted_ids = set()

# ==================================================
# SNAPSHOT MANAGEMENT
# ==================================================

# Stores when each person entered the restricted area
restricted_entry_time = {}

# IDs that already received a snapshot during their current entry
snapshot_taken_ids = set()

# IDs currently inside the restricted area
inside_area_ids = set()

# ==================================================
# RESTRICTED AREA
# ==================================================

restricted_area = None
drawing = False
start_point = None

# ==================================================
# MOUSE CALLBACK
# ==================================================

def draw_restricted_area(event, x, y, flags, param):

    global restricted_area
    global drawing
    global start_point

    # Start drawing
    if event == cv2.EVENT_LBUTTONDOWN:

        drawing = True
        start_point = (x, y)

    # Update rectangle while dragging
    elif event == cv2.EVENT_MOUSEMOVE and drawing:

        x1, y1 = start_point

        restricted_area = (
            min(x1, x),
            min(y1, y),
            max(x1, x),
            max(y1, y)
        )

    # Finish rectangle
    elif event == cv2.EVENT_LBUTTONUP:

        drawing = False

        x1, y1 = start_point

        restricted_area = (
            min(x1, x),
            min(y1, y),
            max(x1, x),
            max(y1, y)
        )

        print("RESTRICTED AREA SET")


# ==================================================
# CREATE WINDOW
# ==================================================

cv2.namedWindow("AI Security Surveillance")

cv2.setMouseCallback(
    "AI Security Surveillance",
    draw_restricted_area
)

# ==================================================
# MAIN LOOP
# ==================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read webcam.")
        break

    # ==================================================
    # YOLO TRACKING
    # ==================================================

    results = model.track(
        frame,
        persist=True,
        classes=[0],
        conf=CONFIDENCE,
        verbose=False
    )

    current_person_ids = set()

    # ==================================================
    # PROCESS DETECTIONS
    # ==================================================

    if results[0].boxes is not None:

        boxes = results[0].boxes

        if boxes.id is not None:

            xyxy = boxes.xyxy.cpu().numpy()
            tracker_ids = boxes.id.int().cpu().tolist()

            for box, yolo_id in zip(
                xyxy,
                tracker_ids
            ):

                x1, y1, x2, y2 = map(int, box)

                # Keep coordinates inside frame
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                # ==================================================
                # FIND PERSON ID
                # ==================================================

                if yolo_id in yolo_to_person:

                    # Existing tracked person
                    person_id = yolo_to_person[yolo_id]

                else:

                    # New person
                    person_id = next_person_id
                    next_person_id += 1

                    yolo_to_person[yolo_id] = person_id

                    print(
                        f"PERSON ENTERED | ID: {person_id}"
                    )

                current_person_ids.add(person_id)

                # ==================================================
                # PERSON IS ACTIVE
                # ==================================================

                active_people[person_id] = {
                    "yolo_id": yolo_id,
                    "missing_frames": 0
                }

                # Person is visible again
                if person_id in missing_people:

                    del missing_people[person_id]

                # ==================================================
                # RESTRICTED AREA CHECK
                # ==================================================

                inside_area = False

                if restricted_area is not None:

                    rx1, ry1, rx2, ry2 = restricted_area

                    # Calculate person's center
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    # Check whether person center is inside area
                    inside_area = (
                        rx1 <= center_x <= rx2
                        and
                        ry1 <= center_y <= ry2
                    )

                # ==================================================
                # RESTRICTED AREA ENTRY
                # ==================================================

                if inside_area:

                    # Red box when inside restricted area
                    box_color = (0, 0, 255)

                    # Detect a fresh entry
                    if person_id not in inside_area_ids:

                        inside_area_ids.add(person_id)

                        # Start the 2-second timer
                        restricted_entry_time[person_id] = time.time()

                        # Allow a new snapshot for this entry
                        snapshot_taken_ids.discard(person_id)

                        # Print restricted-area event
                        print(
                            f"PERSON ID: {person_id} "
                            f"ENTERED RESTRICTED AREA"
                        )

                    # ==================================================
                    # 2-SECOND SNAPSHOT CHECK
                    # ==================================================

                    if (
                        person_id in restricted_entry_time
                        and
                        person_id not in snapshot_taken_ids
                    ):

                        elapsed_time = (
                            time.time()
                            - restricted_entry_time[person_id]
                        )

                        # Person stayed inside for 2 seconds
                        if elapsed_time >= SNAPSHOT_DELAY:

                            # Create timestamp for filename
                            timestamp = datetime.now().strftime(
                                "%Y-%m-%d_%H-%M-%S"
                            )

                            filename = (
                                f"intruder_ID{person_id}_"
                                f"{timestamp}.jpg"
                            )

                            snapshot_path = os.path.join(
                                SNAPSHOT_FOLDER,
                                filename
                            )

                            # Save current camera frame
                            cv2.imwrite(
                                snapshot_path,
                                frame
                            )

                            print(
                                f"SNAPSHOT SAVED | ID: {person_id} | "
                                f"{snapshot_path}"
                            )

                            # Prevent repeated snapshots
                            snapshot_taken_ids.add(person_id)

                            # Mark restricted alert as triggered
                            alerted_ids.add(person_id)

                else:

                    # Green box during normal activity
                    box_color = (0, 255, 0)

                    # ==================================================
                    # PERSON LEFT RESTRICTED AREA
                    # ==================================================

                    if person_id in inside_area_ids:

                        inside_area_ids.discard(person_id)

                        # Remove old timer
                        restricted_entry_time.pop(
                            person_id,
                            None
                        )

                        # Allow a fresh snapshot on next entry
                        snapshot_taken_ids.discard(person_id)

                # ==================================================
                # DRAW PERSON BOX
                # ==================================================

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    2
                )

                # ==================================================
                # DISPLAY PERSON ID
                # ==================================================

                label = f"ID: {person_id}"

                cv2.putText(
                    frame,
                    label,
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    box_color,
                    2
                )

                # ==================================================
                # DISPLAY INTRUDER ALERT
                # ==================================================

                if inside_area:

                    alert_text = "INTRUDER ALERT!"

                    cv2.putText(
                        frame,
                        alert_text,
                        (
                            x1,
                            min(
                                frame.shape[0] - 15,
                                y2 + 30
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

    # ==================================================
    # FIND PEOPLE WHO DISAPPEARED
    # ==================================================

    disappeared_ids = (
        set(active_people.keys())
        - current_person_ids
    )

    for person_id in disappeared_ids:

        # Start missing counter
        if person_id not in missing_people:

            missing_people[person_id] = 0

        # Increase missing frame count
        missing_people[person_id] += 1

        # Update active record
        if person_id in active_people:

            active_people[person_id]["missing_frames"] = (
                missing_people[person_id]
            )

    # ==================================================
    # CHECK FOR PEOPLE WHO HAVE LEFT
    # ==================================================

    people_to_remove = []

    for person_id, missing_frames in missing_people.items():

        if missing_frames >= MAX_MISSING_FRAMES:

            print(
                f"PERSON LEFT | ID: {person_id}"
            )

            people_to_remove.append(person_id)

    # ==================================================
    # REMOVE PEOPLE WHO HAVE LEFT
    # ==================================================

    for person_id in people_to_remove:

        if person_id in active_people:

            del active_people[person_id]

        if person_id in missing_people:

            del missing_people[person_id]

        # Reset restricted-area states
        alerted_ids.discard(person_id)
        inside_area_ids.discard(person_id)
        restricted_entry_time.pop(
            person_id,
            None
        )
        snapshot_taken_ids.discard(person_id)

        # Remove old YOLO tracker mapping
        stale_yolo_ids = [
            yolo_id
            for yolo_id, stable_id in yolo_to_person.items()
            if stable_id == person_id
        ]

        for yolo_id in stale_yolo_ids:

            del yolo_to_person[yolo_id]

    # ==================================================
    # DRAW RESTRICTED AREA
    # ==================================================

    if restricted_area is not None:

        rx1, ry1, rx2, ry2 = restricted_area

        cv2.rectangle(
            frame,
            (rx1, ry1),
            (rx2, ry2),
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            "RESTRICTED AREA",
            (rx1, max(25, ry1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    # ==================================================
    # DISPLAY INSTRUCTION
    # ==================================================

    # Show instruction only before area is selected
    if restricted_area is None:

        cv2.putText(
            frame,
            "Click and drag to select restricted area",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    # ==================================================
    # DISPLAY FRAME
    # ==================================================

    cv2.imshow(
        "AI Security Surveillance",
        frame
    )

    # Press Q to quit
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# ==================================================
# CLEANUP
# ==================================================

cap.release()
cv2.destroyAllWindows()

