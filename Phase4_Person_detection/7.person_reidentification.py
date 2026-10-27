
import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture(0)

CONFIDENCE = 0.5
MAX_MISSING_FRAMES = 120
REID_THRESHOLD = 0.80

next_person_id = 1

active_tracks = {}
last_histograms = {}
missing_people = {}


# Get person's color appearance
def get_histogram(crop):
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist(
        [hsv], [0, 1], None,
        [50, 60],
        [0, 180, 0, 256]
    )

    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

    return hist


while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        conf=CONFIDENCE,
        persist=True,
        verbose=False
    )

    result = results[0]
    current_yolo_ids = set()

    if result.boxes is not None and result.boxes.id is not None:

        tracking_ids = result.boxes.id.int().cpu().tolist()

        for box, yolo_id in zip(result.boxes, tracking_ids):

            class_id = int(box.cls[0])

            if model.names[class_id] != "person":
                continue

            current_yolo_ids.add(yolo_id)

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            crop = frame[y1:y2, x1:x2]
            current_hist = get_histogram(crop)

            # Existing YOLO track
            if yolo_id in active_tracks:

                person_id = active_tracks[yolo_id]

            else:

                person_id = None
                best_score = 0

                # Look for a recently missing person
                for old_id, data in missing_people.items():

                    score = cv2.compareHist(
                        data["histogram"],
                        current_hist,
                        cv2.HISTCMP_CORREL
                    )

                    if score > best_score:
                        best_score = score
                        person_id = old_id

                # Reconnect only with a strong match
                if person_id is not None and best_score >= REID_THRESHOLD:

                    print(f"PERSON RECONNECTED | ID: {person_id}")

                    del missing_people[person_id]

                else:

                    person_id = next_person_id
                    next_person_id += 1

                    print(f"PERSON ENTERED | ID: {person_id}")

                active_tracks[yolo_id] = person_id

            # Save latest appearance
            last_histograms[person_id] = current_hist

            # Draw box and ID
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Person ID: {person_id}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # Detect disappeared people
    for yolo_id, person_id in list(active_tracks.items()):

        if yolo_id not in current_yolo_ids:

            del active_tracks[yolo_id]

            if person_id in last_histograms:

                missing_people[person_id] = {
                    "histogram": last_histograms[person_id],
                    "missing_frames": 1
                }

    # Count missing time
    for person_id in list(missing_people.keys()):

        missing_people[person_id]["missing_frames"] += 1

        if missing_people[person_id]["missing_frames"] >= MAX_MISSING_FRAMES:

            print(f"PERSON LEFT | ID: {person_id}")

            del missing_people[person_id]
            del last_histograms[person_id]

    # Current people count
    people_count = len(active_tracks)

    cv2.putText(
        frame,
        f"People: {people_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("AI Security Surveillance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

