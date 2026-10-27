# ==================================================
# AI SECURITY SURVEILLANCE SYSTEM
# SURVEILLANCE DASHBOARD
# ==================================================

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
# 11. Risk score
# 12. Event logging
# 13. Weapon confirmation
# 14. Surveillance dashboard
# 15. FPS and system status

# ==================================================
# IMPORTS
# ==================================================

import cv2
import os
import time
import torch
import csv

from datetime import datetime
from ultralytics import YOLO


# ==================================================
# SETTINGS
# ==================================================

CONFIDENCE = 0.5

# Person can disappear for this many frames
MAX_MISSING_FRAMES = 100

# Person must stay inside restricted area
# for this many seconds before snapshot
SNAPSHOT_DELAY = 2.0

# Snapshot folder
SNAPSHOT_FOLDER = "snapshots"

# Weapon confirmation settings
WEAPON_CONFIRM_FRAMES = 5
WEAPON_CLEAR_DELAY = 3.0

# Long stay setting
LONG_STAY_DELAY = 10.0

# Risk scores
RESTRICTED_AREA_SCORE = 60
LONG_STAY_SCORE = 30
MULTIPLE_PEOPLE_SCORE = 10
WEAPON_SCORE = 60

# Dashboard settings
DASHBOARD_WIDTH = 1280
DASHBOARD_HEIGHT = 720

# Dashboard panel width
PANEL_WIDTH = 360

# Camera area width
CAMERA_AREA_WIDTH = DASHBOARD_WIDTH - PANEL_WIDTH

# Camera area height
CAMERA_AREA_HEIGHT = DASHBOARD_HEIGHT

# Snapshot folder will be created after PROJECT_ROOT is known.


# ==================================================
# PROJECT ROOT
# ==================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# Use a project-relative snapshot folder so snapshots always go
# to the same place regardless of the working directory.
SNAPSHOT_PATH = os.path.join(
    PROJECT_ROOT,
    SNAPSHOT_FOLDER
)

os.makedirs(
    SNAPSHOT_PATH,
    exist_ok=True
)


# ==================================================
# MODEL PATHS
# ==================================================

PERSON_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "yolo11n.pt"
)

WEAPON_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "weapon_model.pt"
)


# ==================================================
# EVENT LOGGING
# ==================================================

LOG_FOLDER = os.path.join(
    PROJECT_ROOT,
    "logs"
)

LOG_FILE = os.path.join(
    LOG_FOLDER,
    "surveillance_log.csv"
)

# Create logs folder
os.makedirs(
    LOG_FOLDER,
    exist_ok=True
)


# ==================================================
# CREATE CSV FILE
# ==================================================

if not os.path.exists(LOG_FILE):

    with open(
        LOG_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Timestamp",
            "Event",
            "Person ID",
            "Details"
        ])


# ==================================================
# LOG EVENT FUNCTION
# ==================================================

def log_event(
    event,
    person_id="",
    details=""
):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    with open(
        LOG_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            timestamp,
            event,
            person_id,
            details
        ])

    print(
        f"[LOG] {event} | "
        f"ID: {person_id} | "
        f"{details}"
    )

    # Send the same event to the live dashboard history.
    add_dashboard_event(
        event,
        person_id,
        details
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

else:

    DEVICE = "cpu"


# ==================================================
# LOAD MODELS
# ==================================================

print("Loading person detection model...")

person_model = YOLO(
    PERSON_MODEL_PATH
)


print("Loading weapon detection model...")

weapon_model = YOLO(
    WEAPON_MODEL_PATH
)


print("Models loaded successfully.")


# ==================================================
# OPEN WEBCAM
# ==================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")
    exit()


# ==================================================
# GET ORIGINAL CAMERA SIZE
# ==================================================

camera_width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

camera_height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)


# Fallback if camera doesn't report resolution
if camera_width <= 0:

    camera_width = 1280


if camera_height <= 0:

    camera_height = 720


# print(
#     f"Camera resolution: "
#     f"{camera_width}x{camera_height}"
# )


# ==================================================
# PERSON ID MANAGEMENT
# ==================================================

next_person_id = 1

# YOLO tracker ID -> stable person ID
yolo_to_person = {}

# Currently active people
active_people = {}

# Temporarily missing people
missing_people = {}


# ==================================================
# RESTRICTED AREA MANAGEMENT
# ==================================================

# Coordinates are stored in ORIGINAL CAMERA coordinates
restricted_area = None

drawing = False

start_point = None

# People currently inside restricted area
inside_area_ids = set()

# People that triggered restricted alert
alerted_ids = set()


# ==================================================
# SNAPSHOT MANAGEMENT
# ==================================================

# Time when person entered restricted area
restricted_entry_time = {}

# Prevent repeated snapshots
snapshot_taken_ids = set()


# ==================================================
# WEAPON EVENT MANAGEMENT
# ==================================================

weapon_detect_streak = 0

weapon_was_detected = False

weapon_last_seen_time = 0.0

confirmed_weapon_names = set()


# ==================================================
# RISK MANAGEMENT
# ==================================================

# Start with the actual initial state.
# This prevents a NORMAL -> NORMAL log entry at startup.
last_risk_level = "NORMAL"

# ==================================================
# LIVE EVENT HISTORY
# ==================================================

MAX_EVENT_HISTORY = 5

last_event_text = "System started"

last_event_time = datetime.now().strftime(
    "%H:%M:%S"
)

# Events shown live on the dashboard.
# Newest event is stored first.
event_history = [
    {
        "time": last_event_time,
        "event": "SYSTEM_STARTED",
        "details": "Surveillance system started"
    }
]


def add_dashboard_event(
    event,
    person_id="",
    details=""
):

    global last_event_text
    global last_event_time

    event_time = datetime.now().strftime(
        "%H:%M:%S"
    )

    # Include the person ID in the dashboard history when available.
    if person_id != "":

        history_details = (
            f"ID {person_id} | {details}"
        )

    else:

        history_details = details

    event_history.insert(
        0,
        {
            "time": event_time,
            "event": event,
            "details": history_details
        }
    )

    # Keep only the newest events.
    del event_history[MAX_EVENT_HISTORY:]

    last_event_text = event
    last_event_time = event_time


# ==================================================
# CAMERA DISPLAY POSITION
# ==================================================

# These variables describe where the
# actual camera image is placed inside
# the dashboard window.

display_x = 0
display_y = 0

display_width = 0
display_height = 0

scale_x = 1.0
scale_y = 1.0


# ==================================================
# MOUSE CALLBACK
# ==================================================

def draw_restricted_area(
    event,
    x,
    y,
    flags,
    param
):

    global restricted_area
    global drawing
    global start_point

    global display_x
    global display_y

    global display_width
    global display_height

    global scale_x
    global scale_y

    # --------------------------------------------------
    # Only allow mouse interaction inside camera image
    # --------------------------------------------------

    if not (
        display_x <= x < display_x + display_width
        and
        display_y <= y < display_y + display_height
    ):

        return


    # --------------------------------------------------
    # Convert dashboard coordinates
    # into original camera coordinates
    # --------------------------------------------------

    camera_x = int(
        (x - display_x) / scale_x
    )

    camera_y = int(
        (y - display_y) / scale_y
    )


    # Keep coordinates inside camera
    camera_x = max(
        0,
        min(camera_width - 1, camera_x)
    )

    camera_y = max(
        0,
        min(camera_height - 1, camera_y)
    )


    # --------------------------------------------------
    # Start drawing
    # --------------------------------------------------

    if event == cv2.EVENT_LBUTTONDOWN:

        drawing = True

        start_point = (
            camera_x,
            camera_y
        )


    # --------------------------------------------------
    # Update rectangle while dragging
    # --------------------------------------------------

    elif event == cv2.EVENT_MOUSEMOVE:

        if drawing:

            x1, y1 = start_point

            restricted_area = (
                min(x1, camera_x),
                min(y1, camera_y),
                max(x1, camera_x),
                max(y1, camera_y)
            )


    # --------------------------------------------------
    # Finish rectangle
    # --------------------------------------------------

    elif event == cv2.EVENT_LBUTTONUP:

        drawing = False

        x1, y1 = start_point

        restricted_area = (
            min(x1, camera_x),
            min(y1, camera_y),
            max(x1, camera_x),
            max(y1, camera_y)
        )

        print("RESTRICTED AREA SET")

        log_event(
            "RESTRICTED_AREA_SET",
            "",
            "Restricted area selected"
        )


# ==================================================
# CREATE WINDOW
# ==================================================

WINDOW_NAME = "AI Security Surveillance"

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_NAME,
    DASHBOARD_WIDTH,
    DASHBOARD_HEIGHT
)

cv2.setMouseCallback(
    WINDOW_NAME,
    draw_restricted_area
)


# ==================================================
# FPS VARIABLES
# ==================================================

previous_time = time.time()

fps = 0


# ==================================================
# MAIN LOOP
# ==================================================

while True:

    # ==================================================
    # READ CAMERA FRAME
    # ==================================================

    ret, frame = cap.read()

    if not ret:

        print(
            "ERROR: Could not read webcam."
        )

        break


    # ==================================================
    # STORE ORIGINAL FRAME SIZE
    # ==================================================

    camera_height = frame.shape[0]

    camera_width = frame.shape[1]


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


    # Current visible people
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

        xyxy = (
            boxes.xyxy
            .cpu()
            .numpy()
        )

        tracker_ids = (
            boxes.id
            .int()
            .cpu()
            .tolist()
        )

        confidences = (
            boxes.conf
            .cpu()
            .numpy()
        )


        # ==================================================
        # PROCESS EACH PERSON
        # ==================================================

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
                camera_width,
                x2
            )

            y2 = min(
                camera_height,
                y2
            )


            # ==================================================
            # FIND STABLE PERSON ID
            # ==================================================

            if yolo_id in yolo_to_person:

                person_id = yolo_to_person[
                    yolo_id
                ]

            else:

                person_id = next_person_id

                next_person_id += 1

                yolo_to_person[
                    yolo_id
                ] = person_id

                print(
                    f"PERSON ENTERED | "
                    f"ID: {person_id}"
                )

                log_event(
                    "PERSON_ENTERED",
                    person_id,
                    "New person detected"
                )

                last_event_text = (
                    f"PERSON ENTERED | ID {person_id}"
                )

                last_event_time = datetime.now().strftime(
                    "%H:%M:%S"
                )


            # Add current person
            current_person_ids.add(
                person_id
            )


            # ==================================================
            # PERSON ACTIVE
            # ==================================================

            active_people[person_id] = {
                "yolo_id": yolo_id,
                "missing_frames": 0
            }


            # Person visible again
            if person_id in missing_people:

                del missing_people[
                    person_id
                ]


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


                # Check center
                inside_area = (
                    rx1 <= center_x <= rx2
                    and
                    ry1 <= center_y <= ry2
                )


            # ==================================================
            # PERSON INSIDE RESTRICTED AREA
            # ==================================================

            if inside_area:

                box_color = (
                    0,
                    0,
                    255
                )


                # --------------------------------------------------
                # NEW ENTRY
                # --------------------------------------------------

                if person_id not in inside_area_ids:

                    inside_area_ids.add(
                        person_id
                    )

                    restricted_entry_time[
                        person_id
                    ] = time.time()

                    snapshot_taken_ids.discard(
                        person_id
                    )

                    print(
                        f"PERSON ID: "
                        f"{person_id} "
                        f"ENTERED RESTRICTED AREA"
                    )

                    log_event(
                        "RESTRICTED_AREA_ENTRY",
                        person_id,
                        "Person entered restricted area"
                    )

                    last_event_text = (
                        f"RESTRICTED AREA ENTRY | "
                        f"ID {person_id}"
                    )

                    last_event_time = datetime.now().strftime(
                        "%H:%M:%S"
                    )


                # ==================================================
                # SNAPSHOT CHECK
                # ==================================================

                if (
                    person_id in restricted_entry_time
                    and
                    person_id not in snapshot_taken_ids
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

                        timestamp = datetime.now().strftime(
                            "%Y-%m-%d_%H-%M-%S"
                        )


                        filename = (
                            f"intruder_ID{person_id}_"
                            f"{timestamp}.jpg"
                        )


                        snapshot_path = os.path.join(
                            SNAPSHOT_PATH,
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


                        log_event(
                            "SNAPSHOT_SAVED",
                            person_id,
                            f"Snapshot: {snapshot_path}"
                        )


                        snapshot_taken_ids.add(
                            person_id
                        )

                        alerted_ids.add(
                            person_id
                        )

                        last_event_text = (
                            f"SNAPSHOT SAVED | "
                            f"ID {person_id}"
                        )

                        last_event_time = datetime.now().strftime(
                            "%H:%M:%S"
                        )


            # ==================================================
            # PERSON OUTSIDE RESTRICTED AREA
            # ==================================================

            else:

                box_color = (
                    0,
                    255,
                    0
                )


                # --------------------------------------------------
                # LEFT RESTRICTED AREA
                # --------------------------------------------------

                if person_id in inside_area_ids:

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

                    print(
                        f"PERSON ID: "
                        f"{person_id} "
                        f"LEFT RESTRICTED AREA"
                    )

                    log_event(
                        "RESTRICTED_AREA_EXIT",
                        person_id,
                        "Person left restricted area"
                    )

                    last_event_text = (
                        f"RESTRICTED AREA EXIT | "
                        f"ID {person_id}"
                    )

                    last_event_time = datetime.now().strftime(
                        "%H:%M:%S"
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
                (
                    x1,
                    max(
                        25,
                        y1 - 10
                    )
                ),
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
                            camera_height - 15,
                            y2 + 30
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )


    # ==================================================
    # FIND DISAPPEARED PEOPLE
    # ==================================================

    disappeared_ids = (
        set(active_people.keys())
        -
        current_person_ids
    )


    for person_id in disappeared_ids:

        if person_id not in missing_people:

            missing_people[
                person_id
            ] = 0


        missing_people[
            person_id
        ] += 1


        if person_id in active_people:

            active_people[
                person_id
            ]["missing_frames"] = (
                missing_people[
                    person_id
                ]
            )


    # ==================================================
    # CHECK PEOPLE WHO LEFT
    # ==================================================

    people_to_remove = []


    for (
        person_id,
        missing_frames
    ) in missing_people.items():

        if missing_frames >= MAX_MISSING_FRAMES:

            print(
                f"PERSON LEFT | "
                f"ID: {person_id}"
            )


            log_event(
                "PERSON_LEFT",
                person_id,
                "Person left camera view"
            )


            last_event_text = (
                f"PERSON LEFT | ID {person_id}"
            )

            last_event_time = datetime.now().strftime(
                "%H:%M:%S"
            )


            people_to_remove.append(
                person_id
            )


    # ==================================================
    # REMOVE PEOPLE WHO LEFT
    # ==================================================

    for person_id in people_to_remove:

        if person_id in active_people:

            del active_people[
                person_id
            ]


        if person_id in missing_people:

            del missing_people[
                person_id
            ]


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


        # Remove old YOLO mapping
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
    # PEOPLE COUNT
    # ==================================================

    people_count = len(
        current_person_ids
    )


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

    current_weapons = set()


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


            # ==================================================
            # PROCESS EACH WEAPON
            # ==================================================

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


                weapon_name = (
                    weapon_model.names[
                        class_id
                    ]
                )


                weapon_name = str(
                    weapon_name
                ).upper()


                current_weapons.add(
                    weapon_name
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
    # WEAPON CONFIRMATION
    # ==================================================

    weapon_detected_now = (
        weapon_count > 0
    )


    if weapon_detected_now:

        weapon_detect_streak += 1

        weapon_last_seen_time = time.time()


        if (
            not weapon_was_detected
            and
            weapon_detect_streak >= WEAPON_CONFIRM_FRAMES
        ):

            confirmed_weapon_names = (
                current_weapons.copy()
            )


            weapon_list = ", ".join(
                sorted(
                    confirmed_weapon_names
                )
            )


            log_event(
                "WEAPON_CONFIRMED",
                "",
                f"Confirmed: {weapon_list}"
            )


            weapon_was_detected = True


            last_event_text = (
                f"WEAPON CONFIRMED | "
                f"{weapon_list}"
            )

            last_event_time = datetime.now().strftime(
                "%H:%M:%S"
            )


    else:

        weapon_detect_streak = 0


        if (
            weapon_was_detected
            and
            time.time()
            -
            weapon_last_seen_time
            >= WEAPON_CLEAR_DELAY
        ):

            log_event(
                "WEAPON_CLEARED",
                "",
                f"No weapon detected for "
                f"{WEAPON_CLEAR_DELAY:.0f} seconds"
            )


            weapon_was_detected = False

            confirmed_weapon_names.clear()


            last_event_text = (
                "WEAPON CLEARED"
            )

            last_event_time = datetime.now().strftime(
                "%H:%M:%S"
            )


    # ==================================================
    # RISK ANALYSIS ENGINE
    # ==================================================

    risk_score = 0


    # Restricted area
    if inside_area_ids:

        risk_score += (
            RESTRICTED_AREA_SCORE
        )


    # Long stay
    long_stay_detected = False


    for person_id in inside_area_ids:

        if person_id in restricted_entry_time:

            if (
                time.time()
                -
                restricted_entry_time[
                    person_id
                ]
                >= LONG_STAY_DELAY
            ):

                long_stay_detected = True

                break


    if long_stay_detected:

        risk_score += (
            LONG_STAY_SCORE
        )


    # Multiple people
    if people_count >= 2:

        risk_score += (
            MULTIPLE_PEOPLE_SCORE
        )


    # Confirmed weapon
    if weapon_was_detected:

        risk_score += (
            WEAPON_SCORE
        )


    # ==================================================
    # RISK LEVEL
    # ==================================================

    if risk_score >= 60:

        risk_level = "CRITICAL"

    elif risk_score >= 30:

        risk_level = "SUSPICIOUS"

    else:

        risk_level = "NORMAL"


    # ==================================================
    # RISK LEVEL CHANGE LOGGING
    # ==================================================

    if risk_level != last_risk_level:

        log_event(
            "RISK_LEVEL_CHANGED",
            "",
            f"Risk: {risk_level} | "
            f"Score: {risk_score}"
        )


        print(
            f"RISK LEVEL: "
            f"{risk_level} | "
            f"SCORE: {risk_score}"
        )


        last_risk_level = risk_level


        # log_event() already updates the live event history.


    # ==================================================
    # FPS
    # ==================================================

    current_time = time.time()


    fps = 1 / max(
        current_time - previous_time,
        0.001
    )


    previous_time = current_time


    # ==================================================
    # CREATE DASHBOARD CANVAS
    # ==================================================

    dashboard = cv2.UMat(
        DASHBOARD_HEIGHT,
        DASHBOARD_WIDTH,
        cv2.CV_8UC3
    ).get()


    # ==================================================
    # CALCULATE CAMERA DISPLAY SIZE
    # ==================================================

    camera_ratio = (
        camera_width / camera_height
    )


    available_width = (
        CAMERA_AREA_WIDTH
    )

    available_height = (
        CAMERA_AREA_HEIGHT
    )


    # Fit camera while preserving aspect ratio
    if (
        available_width
        /
        available_height
        <
        camera_ratio
    ):

        display_width = available_width

        display_height = int(
            display_width
            /
            camera_ratio
        )

    else:

        display_height = available_height

        display_width = int(
            display_height
            *
            camera_ratio
        )


    # Center camera inside camera area
    display_x = int(
        (
            CAMERA_AREA_WIDTH
            -
            display_width
        )
        /
        2
    )


    display_y = int(
        (
            CAMERA_AREA_HEIGHT
            -
            display_height
        )
        /
        2
    )


    # Scale factors
    scale_x = (
        display_width
        /
        camera_width
    )

    scale_y = (
        display_height
        /
        camera_height
    )


    # ==================================================
    # RESIZE CAMERA WITHOUT DISTORTION
    # ==================================================

    display_frame = cv2.resize(
        frame,
        (
            display_width,
            display_height
        ),
        interpolation=cv2.INTER_AREA
    )


    # ==================================================
    # PUT CAMERA INTO DASHBOARD
    # ==================================================

    dashboard[
        display_y:
        display_y + display_height,
        display_x:
        display_x + display_width
    ] = display_frame


    # ==================================================
    # DRAW CAMERA BORDER
    # ==================================================

    cv2.rectangle(
        dashboard,
        (
            display_x,
            display_y
        ),
        (
            display_x + display_width - 1,
            display_y + display_height - 1
        ),
        (70, 70, 70),
        2
    )


    # ==================================================
    # DRAW RESTRICTED AREA
    # ==================================================

    if restricted_area is not None:

        rx1, ry1, rx2, ry2 = (
            restricted_area
        )


        # Convert original camera coordinates
        # to dashboard coordinates

        drx1 = int(
            display_x
            +
            rx1 * scale_x
        )

        dry1 = int(
            display_y
            +
            ry1 * scale_y
        )

        drx2 = int(
            display_x
            +
            rx2 * scale_x
        )

        dry2 = int(
            display_y
            +
            ry2 * scale_y
        )


        cv2.rectangle(
            dashboard,
            (
                drx1,
                dry1
            ),
            (
                drx2,
                dry2
            ),
            (0, 0, 255),
            2
        )


        cv2.putText(
            dashboard,
            "RESTRICTED AREA",
            (
                drx1,
                max(
                    25,
                    dry1 - 10
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2
        )


    # ==================================================
    # CAMERA LABEL
    # ==================================================

    cv2.rectangle(
        dashboard,
        (
            display_x + 10,
            display_y + 10
        ),
        (
            display_x + 160,
            display_y + 45
        ),
        (20, 20, 20),
        -1
    )


    cv2.putText(
        dashboard,
        "LIVE CAMERA",
        (
            display_x + 20,
            display_y + 34
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ==================================================
    # DASHBOARD PANEL
    # ==================================================

    panel_x = CAMERA_AREA_WIDTH

    panel_y = 0

    panel_right = DASHBOARD_WIDTH

    panel_bottom = DASHBOARD_HEIGHT


    # Dark dashboard background
    dashboard[
        panel_y:panel_bottom,
        panel_x:panel_right
    ] = (
        25,
        25,
        25
    )


    # ==================================================
    # PANEL HEADER
    # ==================================================

    cv2.putText(
        dashboard,
        "AI SECURITY",
        (
            panel_x + 25,
            45
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )


    cv2.putText(
        dashboard,
        "SURVEILLANCE",
        (
            panel_x + 25,
            80
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (180, 180, 180),
        2
    )


    # Header separator
    cv2.line(
        dashboard,
        (
            panel_x + 20,
            100
        ),
        (
            panel_right - 20,
            100
        ),
        (80, 80, 80),
        1
    )


    # ==================================================
    # PEOPLE
    # ==================================================

    cv2.putText(
        dashboard,
        "PEOPLE",
        (
            panel_x + 25,
            140
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (160, 160, 160),
        1
    )


    cv2.putText(
        dashboard,
        str(people_count),
        (
            panel_x + 25,
            175
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )


    # ==================================================
    # WEAPONS
    # ==================================================

    cv2.putText(
        dashboard,
        "WEAPONS",
        (
            panel_x + 180,
            140
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (160, 160, 160),
        1
    )


    weapon_display_count = len(
        confirmed_weapon_names
    )


    cv2.putText(
        dashboard,
        str(weapon_display_count),
        (
            panel_x + 180,
            175
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )


    # ==================================================
    # RISK
    # ==================================================

    cv2.putText(
        dashboard,
        "RISK LEVEL",
        (
            panel_x + 25,
            220
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (160, 160, 160),
        1
    )


    if risk_level == "CRITICAL":

        risk_color = (
            0,
            0,
            255
        )

    elif risk_level == "SUSPICIOUS":

        risk_color = (
            0,
            165,
            255
        )

    else:

        risk_color = (
            0,
            255,
            0
        )


    cv2.putText(
        dashboard,
        risk_level,
        (
            panel_x + 25,
            260
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        risk_color,
        2
    )


    # ==================================================
    # RISK SCORE
    # ==================================================

    cv2.putText(
        dashboard,
        f"Risk Score: {risk_score}",
        (
            panel_x + 25,
            295
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (220, 220, 220),
        2
    )


    # ==================================================
    # RESTRICTED AREA STATUS
    # ==================================================

    cv2.putText(
        dashboard,
        "RESTRICTED AREA",
        (
            panel_x + 25,
            340
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (160, 160, 160),
        1
    )


    if restricted_area is not None:

        restricted_status = "ACTIVE"

        restricted_color = (
            0,
            0,
            255
        )

    else:

        restricted_status = "NOT SET"

        restricted_color = (
            180,
            180,
            180
        )


    cv2.putText(
        dashboard,
        restricted_status,
        (
            panel_x + 25,
            375
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        restricted_color,
        2
    )


    # ==================================================
    # FPS
    # ==================================================

    # cv2.putText(
    #     dashboard,
    #     f"FPS: {fps:.1f}",
    #     (
    #         panel_x + 180,
    #         375
    #     ),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.65,
    #     (220, 220, 220),
    #     2
    # )


    # ==================================================
    # CURRENT WEAPONS
    # ==================================================

    cv2.putText(
        dashboard,
        "CONFIRMED WEAPONS",
        (
            panel_x + 25,
            420
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (160, 160, 160),
        1
    )


    if weapon_was_detected:

        weapon_text = ", ".join(
            sorted(
                confirmed_weapon_names
            )
        )

        if len(weapon_text) > 28:

            weapon_text = (
                weapon_text[:25]
                +
                "..."
            )

        weapon_text_color = (
            0,
            0,
            255
        )

    else:

        weapon_text = "None"

        weapon_text_color = (
            180,
            180,
            180
        )


    cv2.putText(
        dashboard,
        weapon_text,
        (
            panel_x + 25,
            450
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        weapon_text_color,
        2
    )


    # ==================================================
    # STATUS
    # ==================================================

    cv2.putText(
        dashboard,
        "SYSTEM STATUS",
        (
            panel_x + 25,
            495
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (160, 160, 160),
        1
    )


    if weapon_was_detected:

        system_status = "WEAPON CONFIRMED"

        status_color = (
            0,
            0,
            255
        )

    elif inside_area_ids:

        system_status = "INTRUDER IN AREA"

        status_color = (
            0,
            0,
            255
        )

    elif people_count > 0:

        system_status = "PERSON DETECTED"

        status_color = (
            0,
            255,
            0
        )

    else:

        system_status = "MONITORING"

        status_color = (
            180,
            180,
            180
        )


    cv2.putText(
        dashboard,
        system_status,
        (
            panel_x + 25,
            530
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        status_color,
        2
    )


    # ==================================================
    # LIVE EVENT HISTORY
    # ==================================================

    cv2.putText(
        dashboard,
        "EVENT HISTORY",
        (
            panel_x + 25,
            570
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (160, 160, 160),
        1
    )

    history_y = 590

    for index, event in enumerate(
        event_history
    ):

        row_y = (
            history_y
            +
            (index * 20)
        )

        event_time = event["time"]
        event_name = event["event"]
        details = event["details"]

        # Build one compact line so several events fit
        # inside the fixed 720p dashboard.
        history_text = (
            f"{event_time}  {event_name}"
        )

        if details:

            history_text += (
                f" | {details}"
            )

        if len(history_text) > 45:

            history_text = (
                history_text[:42]
                +
                "..."
            )

        cv2.putText(
            dashboard,
            history_text,
            (
                panel_x + 20,
                row_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (220, 220, 220),
            1
        )


    # ==================================================
    # DASHBOARD INSTRUCTIONS
    # ==================================================

    cv2.line(
        dashboard,
        (
            panel_x + 20,
            735 - 85
        ),
        (
            panel_right - 20,
            735 - 85
        ),
        (80, 80, 80),
        1
    )

    cv2.putText(
        dashboard,
        "Drag inside camera to set area",
        (
            panel_x + 20,
            680
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (160, 160, 160),
        1
    )

    cv2.putText(
        dashboard,
        "Q = Quit",
        (
            panel_x + 20,
            705
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (160, 160, 160),
        1
    )


    # ==================================================
    # CURRENT CLOCK
    # ==================================================

    current_clock = datetime.now().strftime(
        "%H:%M:%S"
    )

    cv2.putText(
        dashboard,
        current_clock,
        (
            panel_x + 240,
            45
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (170, 170, 170),
        1
    )


    # ==================================================
    # DISPLAY DASHBOARD
    # ==================================================

    cv2.imshow(
        WINDOW_NAME,
        dashboard
    )


    # ==================================================
    # PRESS Q TO EXIT
    # ==================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q") or key == ord("Q"):

        break


# ==================================================
# CLEANUP
# ==================================================

cap.release()

cv2.destroyAllWindows()