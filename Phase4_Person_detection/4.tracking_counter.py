import cv2
from ultralytics import YOLO

# Load the pre-trained YOLO model
model = YOLO("yolo11n.pt")

# Open the default webcam
cap = cv2.VideoCapture(0)

while True:

    # Capture one frame from the webcam
    ret, frame = cap.read()

    # Stop if the camera fails
    if not ret:
        break

    # Run YOLO tracking on the current frame
    # persist=True keeps the tracking IDs between frames
    results = model.track(frame, conf=0.5, persist=True,verbose=False)

    # Get the first result
    result = results[0]

    # Create an empty set to store unique person IDs
    person_ids = set()

    # Check if tracking IDs are available
    if result.boxes.id is not None:

        # Go through every detected object
        for box, track_id in zip(result.boxes, result.boxes.id):

            # Get the object's class number
            class_id = int(box.cls[0])

            # Convert class number to class name
            class_name = model.names[class_id]

            # Only count people
            if class_name == "person":

                # Add the tracking ID to the set
                person_ids.add(int(track_id))

    # Count the unique tracked people
    person_count = len(person_ids)

    # Draw bounding boxes and tracking IDs
    annotated_frame = result.plot()

    # Display the current number of people
    cv2.putText(
        annotated_frame,
        f"People: {person_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Show the processed frame
    cv2.imshow("YOLO Tracking Counter", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()