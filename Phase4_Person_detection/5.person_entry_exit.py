import cv2
from ultralytics import YOLO

# Load the pre-trained YOLO model
model = YOLO("yolo11n.pt")

# Open the default webcam
cap = cv2.VideoCapture(0)

# Store the IDs that were visible in the previous frame
previous_ids = set()

while True:

    # Capture one frame from the webcam
    ret, frame = cap.read()

    # Stop if the camera fails
    if not ret:
        break

    # Run YOLO tracking
    results = model.track(frame, conf=0.5, persist=True, verbose=False)

    # Get the first result
    result = results[0]

    # Store the IDs detected in the current frame
    current_ids = set()

    # Check whether tracking IDs are available
    if result.boxes.id is not None:

        # Go through every detected object
        for box, track_id in zip(result.boxes, result.boxes.id):

            # Get the object's class number
            class_id = int(box.cls[0])

            # Convert class number into its name
            class_name = model.names[class_id]

            # We only care about people
            if class_name == "person":

                # Add this person's tracking ID
                current_ids.add(int(track_id))

    # Find people who appeared for the first time
    entered = current_ids - previous_ids

    # Find people who disappeared
    left = previous_ids - current_ids

    # Print entry messages
    for person_id in entered:
        print(f"PERSON ENTERED | ID: {person_id}")

    # Print exit messages
    for person_id in left:
        print(f"PERSON LEFT | ID: {person_id}")

    # Remember the current IDs for the next frame
    previous_ids = current_ids.copy()

    # Draw detections and tracking IDs
    annotated_frame = result.plot()

    # Display current number of people
    cv2.putText(
        annotated_frame,
        f"People: {len(current_ids)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Show the video
    cv2.imshow("Security Tracking", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release webcam
cap.release()

# Close OpenCV windows
cv2.destroyAllWindows()