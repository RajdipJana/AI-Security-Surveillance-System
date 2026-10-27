
import cv2
import os
import csv
import time
from datetime import datetime
from ultralytics import YOLO

# ==================================================
# SETTINGS
# ==================================================

CONFIDENCE = 0.5

# Frames a person can be missing before being considered LEFT
MAX_MISSING_FRAMES = 100

# Person must remain inside restricted area for this time
# before a snapshot is captured
SNAPSHOT_DELAY = 2.0

# Folder for snapshots
SNAPSHOT_FOLDER = "snapshots"

# Folder for event logs
LOG_FOLDER = "logs"

# CSV log file
LOG_FILE = os.path.join(
    LOG_FOLDER,
    "surveillance_log.csv"
)

# Create required folders automatically
os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)

# ==================================================
# EVENT LOGGER
# ==================================================

def log_event(event, person_id, details):
    # Get current date and time
    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    # Check whether CSV file already exists
    file_exists = os.path.exists(LOG_FILE)

    # Open CSV file in append mode
    with open(
        LOG_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        # Create header for the first entry
        if not file_exists:
            writer.writerow([
                "Timestamp",
                "Event",
                "Person ID",
                "Details"
            ])

        # Write event information
        writer.writerow([
            timestamp,
            event,
            person_id,
            details
        ])


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

# ==================================================
# RESTRICTED AREA
# ==================================================

restricted_area = None
drawing = False
start_point = None

# ==================================================
# SNAPSHOT MANAGEMENT
# ==================================================

# Time when each person entered restricted area
restricted_entry_time = {}

# IDs that already received a snapshot
snapshot_taken_ids = set()

# IDs currently inside restricted area
inside_area_ids = set()

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

            tracker_ids = (
                boxes.id.int()
                .cpu()
                .tolist()
            )

            for box, yolo_id in zip(
                xyxy,
                tracker_ids
            ):

                x1, y1, x2, y2 = map(
                    int,
                    box
                )

                # Keep coordinates inside frame
                x1 = max(0, x1)
                y1 = max(0, y1)

                x2 = min(
                    frame.shape[1],
                    x2
                )

                y2 = min(
                    frame.shape[0],
                    y2
                )

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

                    # Save event to CSV
                    log_event(
                        "PERSON_ENTERED",
                        person_id,
                        "Person detected by surveillance system"
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

                    # Person has just entered restricted area
                    if person_id not in inside_area_ids:

                        inside_area_ids.add(person_id)

                        # Start 2-second confirmation timer
                        restricted_entry_time[person_id] = time.time()

                        # Allow a new snapshot on this entry
                        snapshot_taken_ids.discard(person_id)

                        print(
                            f"PERSON ID: {person_id} "
                            f"ENTERED RESTRICTED AREA"
                        )

                        # Save restricted-area event
                        log_event(
                            "RESTRICTED_AREA_ENTRY",
                            person_id,
                            "Person entered restricted area"
                        )

                    # ==================================================
                    # SNAPSHOT CONFIRMATION
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

                        # Confirm person after 2 seconds
                        if elapsed_time >= SNAPSHOT_DELAY:

                            timestamp = datetime.now().strftime(
                                "%Y-%m-%d_%H-%M-%S"
                            )

                            snapshot_filename = (
                                f"ID_{person_id}_"
                                f"{timestamp}.jpg"
                            )

                            snapshot_path = os.path.join(
                                SNAPSHOT_FOLDER,
                                snapshot_filename
                            )

                            # Save current frame
                            cv2.imwrite(
                                snapshot_path,
                                frame
                            )

                            print(
                                f"SNAPSHOT SAVED | "
                                f"ID: {person_id} | "
                                f"{snapshot_path}"
                            )

                            # Save snapshot event
                            log_event(
                                "SNAPSHOT_SAVED",
                                person_id,
                                "Restricted-area presence confirmed for 2 seconds"
                            )

                            # Mark snapshot as completed
                            snapshot_taken_ids.add(
                                person_id
                            )

                else:

                    # Person has left restricted area
                    if person_id in inside_area_ids:

                        inside_area_ids.discard(
                            person_id
                        )

                        restricted_entry_time.pop(
                            person_id,
                            None
                        )

                        # Allow another snapshot if
                        # the person enters again
                        snapshot_taken_ids.discard(
                            person_id
                        )

                        # Save restricted-area exit event
                        log_event(
                            "RESTRICTED_AREA_EXIT",
                            person_id,
                            "Person left restricted area"
                        )

                # ==================================================
                # SET BOX COLOR
                # ==================================================

                if inside_area:

                    # Red box inside restricted area
                    box_color = (0, 0, 255)

                else:

                    # Green box during normal activity
                    box_color = (0, 255, 0)

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

            # Save person-left event
            log_event(
                "PERSON_LEFT",
                person_id,
                "Person disappeared from camera view"
            )

            people_to_remove.append(
                person_id
            )

    # ==================================================
    # REMOVE PEOPLE WHO HAVE LEFT
    # ==================================================

    for person_id in people_to_remove:

        if person_id in active_people:

            del active_people[person_id]

        if person_id in missing_people:

            del missing_people[person_id]

        # Reset restricted-area state
        inside_area_ids.discard(
            person_id
        )

        restricted_entry_time.pop(
            person_id,
            None
        )

        snapshot_taken_ids.discard(
            person_id
        )

        # Remove old YOLO tracker mapping
        stale_yolo_ids = [
            yolo_id
            for yolo_id, stable_id
            in yolo_to_person.items()
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

