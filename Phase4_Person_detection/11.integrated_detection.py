# ==================================================
# AI SECURITY SURVEILLANCE SYSTEM
# INTEGRATED DETECTION
# ==================================================
#
# Features:
# 1. Person detection
# 2. Person tracking
# 3. Stable person IDs
# 4. Person entered / left detection
# 5. Restricted area detection
# 6. Intruder alert
# 7. 2-second restricted-area snapshot
# 8. Gun detection
# 9. Knife detection
# 10. People and weapon counters
# 11. FPS display
#
# ==================================================

import cv2
import os
import time
import torch

from datetime import datetime
from ultralytics import YOLO


# ==================================================
# SETTINGS
# ==================================================

CONFIDENCE = 0.5

# Person can disappear for this many frames
# before being considered LEFT
MAX_MISSING_FRAMES = 100

# Person must stay inside restricted area
# for this many seconds before snapshot
SNAPSHOT_DELAY = 2.0

# Snapshot folder
SNAPSHOT_FOLDER = "snapshots"

# Create snapshot folder if needed
os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)


# ==================================================
# PROJECT ROOT
# ==================================================

# Get the folder containing this Python file
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# Model paths
PERSON_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "yolo11n.pt"
)

WEAPON_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "weapon_model.pt"
)


# ==================================================
# CHECK MODEL FILES
# ==================================================

if not os.path.exists(PERSON_MODEL_PATH):
    print("ERROR: yolo11n.pt not found.")
    print(PERSON_MODEL_PATH)
    exit()

if not os.path.exists(WEAPON_MODEL_PATH):
    print("ERROR: weapon_model.pt not found.")
    print(WEAPON_MODEL_PATH)
    exit()


# ==================================================
# SELECT DEVICE
# ==================================================

if torch.cuda.is_available():
    DEVICE = 0
    # print("GPU DETECTED")
    # print(torch.cuda.get_device_name(0))    #this is the code for display the gpu status and name-----------------
else:
    DEVICE = "cpu"
    # print("CUDA NOT AVAILABLE")
    # print("Using CPU")


# ==================================================
# LOAD MODELS
# ==================================================

print("Loading person detection model...")
person_model = YOLO(PERSON_MODEL_PATH)

print("Loading weapon detection model...")
weapon_model = YOLO(WEAPON_MODEL_PATH)

print("Models loaded successfully.")


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

# Our own stable person ID
next_person_id = 1

# YOLO tracker ID -> our stable person ID
yolo_to_person = {}

# Currently active people
active_people = {}

# People temporarily missing
missing_people = {}


# ==================================================
# RESTRICTED AREA MANAGEMENT
# ==================================================

restricted_area = None

drawing = False

start_point = None

# IDs currently inside restricted area
inside_area_ids = set()

# IDs that already triggered restricted alert
alerted_ids = set()


# ==================================================
# SNAPSHOT MANAGEMENT
# ==================================================

# Stores the time when each person entered
# the restricted area
restricted_entry_time = {}

# Prevents multiple snapshots during
# the same restricted-area entry
snapshot_taken_ids = set()


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
# FPS VARIABLES
# ==================================================

previous_time = time.time()


# ==================================================
# MAIN LOOP
# ==================================================

while True:

    # --------------------------------------------------
    # READ CAMERA FRAME
    # --------------------------------------------------

    ret, frame = cap.read()

    if not ret:

        print("ERROR: Could not read webcam.")

        break


    # ==================================================
    # PERSON TRACKING
    # ==================================================

    person_results = person_model.track(
        frame,
        persist=True,
        classes=[0],
        conf=CONFIDENCE,
        device=DEVICE,
        verbose=False
    )


    # Current people visible in this frame
    current_person_ids = set()


    # ==================================================
    # PROCESS PERSON DETECTIONS
    # ==================================================

    if (
        person_results[0].boxes is not None
        and
        person_results[0].boxes.id is not None
    ):

        boxes = person_results[0].boxes

        xyxy = boxes.xyxy.cpu().numpy()

        tracker_ids = (
            boxes.id.int()
            .cpu()
            .tolist()
        )

        confidences = (
            boxes.conf
            .cpu()
            .numpy()
        )


        # --------------------------------------------------
        # PROCESS EACH PERSON
        # --------------------------------------------------

        for box, yolo_id, confidence in zip(
            xyxy,
            tracker_ids,
            confidences
        ):

            x1, y1, x2, y2 = map(
                int,
                box
            )


            # Keep coordinates inside frame
            x1 = max(
                0,
                x1
            )

            y1 = max(
                0,
                y1
            )

            x2 = min(
                frame.shape[1],
                x2
            )

            y2 = min(
                frame.shape[0],
                y2
            )


            # ==================================================
            # FIND STABLE PERSON ID
            # ==================================================

            if yolo_id in yolo_to_person:

                person_id = yolo_to_person[yolo_id]

            else:

                # New person detected
                person_id = next_person_id

                next_person_id += 1

                yolo_to_person[yolo_id] = person_id

                print(
                    f"PERSON ENTERED | ID: {person_id}"
                )
                


            # Add to current people
            current_person_ids.add(
                person_id
            )


            # ==================================================
            # PERSON IS ACTIVE
            # ==================================================

            active_people[person_id] = {
                "yolo_id": yolo_id,
                "missing_frames": 0
            }


            # Person visible again
            if person_id in missing_people:

                del missing_people[person_id]


            # ==================================================
            # RESTRICTED AREA CHECK
            # ==================================================

            inside_area = False

            if restricted_area is not None:

                rx1, ry1, rx2, ry2 = (
                    restricted_area
                )

                # Person center
                center_x = (
                    x1 + x2
                ) // 2

                center_y = (
                    y1 + y2
                ) // 2


                # Check if center is inside area
                inside_area = (
                    rx1 <= center_x <= rx2
                    and
                    ry1 <= center_y <= ry2
                )


            # ==================================================
            # PERSON INSIDE RESTRICTED AREA
            # ==================================================

            if inside_area:

                # Red person box
                box_color = (
                    0,
                    0,
                    255
                )


                # --------------------------------------------------
                # NEW RESTRICTED-AREA ENTRY
                # --------------------------------------------------

                if person_id not in inside_area_ids:

                    inside_area_ids.add(
                        person_id
                    )

                    # Start 2-second timer
                    restricted_entry_time[
                        person_id
                    ] = time.time()

                    # Allow fresh snapshot
                    snapshot_taken_ids.discard(
                        person_id
                    )


                    print(
                        f"PERSON ID: {person_id} "
                        f"ENTERED RESTRICTED AREA"
                    )


                # ==================================================
                # 2-SECOND SNAPSHOT CHECK
                # ==================================================

                if (
                    person_id
                    in restricted_entry_time
                    and
                    person_id
                    not in snapshot_taken_ids
                ):

                    elapsed_time = (
                        time.time()
                        -
                        restricted_entry_time[
                            person_id
                        ]
                    )


                    # Person stayed for 2 seconds
                    if elapsed_time >= SNAPSHOT_DELAY:

                        # Timestamp
                        timestamp = (
                            datetime.now()
                            .strftime(
                                "%Y-%m-%d_%H-%M-%S"
                            )
                        )


                        # Snapshot filename
                        filename = (
                            f"intruder_ID{person_id}_"
                            f"{timestamp}.jpg"
                        )


                        snapshot_path = os.path.join(
                            SNAPSHOT_FOLDER,
                            filename
                        )


                        # Save frame
                        cv2.imwrite(
                            snapshot_path,
                            frame
                        )


                        print(
                            f"SNAPSHOT SAVED | "
                            f"ID: {person_id} | "
                            f"{snapshot_path}"
                        )


                        # Prevent repeated snapshots
                        snapshot_taken_ids.add(
                            person_id
                        )


                        # Mark alert
                        alerted_ids.add(
                            person_id
                        )


            # ==================================================
            # PERSON OUTSIDE RESTRICTED AREA
            # ==================================================

            else:

                # Green person box
                box_color = (
                    0,
                    255,
                    0
                )


                # --------------------------------------------------
                # PERSON LEFT RESTRICTED AREA
                # --------------------------------------------------

                if person_id in inside_area_ids:

                    inside_area_ids.discard(
                        person_id
                    )


                    # Remove old timer
                    restricted_entry_time.pop(
                        person_id,
                        None
                    )


                    # Allow fresh snapshot
                    snapshot_taken_ids.discard(
                        person_id
                    )


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
            # PERSON LABEL
            # ==================================================

            person_label = (
                f"ID: {person_id} | "
                f"Person {confidence:.2f}"
            )


            cv2.putText(
                frame,
                person_label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2
            )


            # ==================================================
            # INTRUDER ALERT
            # ==================================================

            if inside_area:

                cv2.putText(
                    frame,
                    "INTRUDER ALERT!",
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
        -
        current_person_ids
    )


    for person_id in disappeared_ids:

        # Start missing counter
        if person_id not in missing_people:

            missing_people[person_id] = 0


        # Increase missing frames
        missing_people[person_id] += 1


        # Update active record
        if person_id in active_people:

            active_people[
                person_id
            ]["missing_frames"] = (
                missing_people[person_id]
            )


    # ==================================================
    # CHECK PEOPLE WHO HAVE LEFT
    # ==================================================

    people_to_remove = []


    for (
        person_id,
        missing_frames
    ) in missing_people.items():

        if missing_frames >= MAX_MISSING_FRAMES:

            print(
                f"PERSON LEFT | ID: {person_id}"
            )

            people_to_remove.append(
                person_id
            )


    # ==================================================
    # REMOVE PEOPLE WHO LEFT
    # ==================================================

    for person_id in people_to_remove:

        # Remove active person
        if person_id in active_people:

            del active_people[
                person_id
            ]


        # Remove missing record
        if person_id in missing_people:

            del missing_people[
                person_id
            ]


        # Reset restricted-area state
        alerted_ids.discard(
            person_id
        )

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
            for (
                yolo_id,
                stable_id
            ) in yolo_to_person.items()
            if stable_id == person_id
        ]


        for yolo_id in stale_yolo_ids:

            del yolo_to_person[
                yolo_id
            ]


    # ==================================================
    # WEAPON DETECTION
    # ==================================================

    weapon_results = weapon_model(
        frame,
        conf=CONFIDENCE,
        device=DEVICE,
        verbose=False
    )


    weapon_count = 0


    # ==================================================
    # PROCESS WEAPONS
    # ==================================================

    if weapon_results[0].boxes is not None:

        weapon_boxes = (
            weapon_results[0].boxes
        )


        if len(weapon_boxes) > 0:

            weapon_count = len(
                weapon_boxes
            )


            # --------------------------------------------------
            # PROCESS EACH WEAPON
            # --------------------------------------------------

            for weapon_box in weapon_boxes:

                wx1, wy1, wx2, wy2 = map(
                    int,
                    weapon_box.xyxy[0]
                    .cpu()
                    .numpy()
                )


                weapon_confidence = float(
                    weapon_box.conf[0]
                    .cpu()
                    .item()
                )


                class_id = int(
                    weapon_box.cls[0]
                    .cpu()
                    .item()
                )


                # Get weapon class name
                weapon_name = (
                    weapon_model.names[
                        class_id
                    ]
                )


                # Convert to uppercase
                weapon_name = (
                    str(weapon_name)
                    .upper()
                )


                # ==================================================
                # DRAW WEAPON BOX
                # ==================================================

                cv2.rectangle(
                    frame,
                    (wx1, wy1),
                    (wx2, wy2),
                    (0, 0, 255),
                    2
                )


                # ==================================================
                # WEAPON LABEL
                # ==================================================

                weapon_label = (
                    f"{weapon_name} "
                    f"{weapon_confidence:.2f}"
                )


                cv2.putText(
                    frame,
                    weapon_label,
                    (
                        wx1,
                        max(
                            25,
                            wy1 - 10
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )


    # ==================================================
    # WEAPON ALERT
    # ==================================================

    if weapon_count > 0:

        cv2.putText(
            frame,
            "!!! WEAPON DETECTED !!!",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )
        # print("WEAPON DETECTED...")


    # ==================================================
    # DRAW RESTRICTED AREA
    # ==================================================

    if restricted_area is not None:

        rx1, ry1, rx2, ry2 = (
            restricted_area
        )


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
            (
                rx1,
                max(
                    25,
                    ry1 - 10
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )


    # ==================================================
    # DISPLAY INSTRUCTION
    # ==================================================

    if restricted_area is None:

        cv2.putText(
            frame,
            "Click and drag to select restricted area",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (45, 19, 22),
            2
        )


    # ==================================================
    # CALCULATE FPS
    # ==================================================

    current_time = time.time()

    fps = 1 / max(
        current_time - previous_time,
        0.001
    )

    previous_time = current_time


    # ==================================================
    # DISPLAY INFORMATION
    # ==================================================

    people_count = len(
        current_person_ids
    )

    if(people_count>0):
        cv2.putText(
                frame,
                f"People: {people_count}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (45, 19, 22),
                2
            )



    # cv2.putText(
    #     frame,
    #     f"People: {people_count}",
    #     (20, 105),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.7,
    #     (45, 19, 22),
    #     2
    # )

    if (weapon_count>0):
        cv2.putText(
                frame,
                f"Weapons: {weapon_count}",
                (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (45, 19, 22),
                2
            )
    
    # cv2.putText(
    #     frame,
    #     f"Weapons: {weapon_count}",
    #     (20, 135),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.7,
    #     (45, 19, 22),
    #     2
    # )


    # cv2.putText(
    #     frame,
    #     f"FPS: {fps:.1f}",
    #     (20, 165),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.7,
    #     (255, 255, 255),
    #     2
    # )


    # ==================================================
    # DISPLAY FRAME
    # ==================================================

    cv2.imshow(
        "AI Security Surveillance",
        frame
    )


    # ==================================================
    # PRESS Q TO EXIT
    # ==================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key== ord("Q"):

        break



# ==================================================
# CLEANUP
# ==================================================

cap.release()

cv2.destroyAllWindows()
