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

    # Run YOLO on the current frame
    results = model(frame, conf=0.5)

    # Get the first result
    result = results[0]

    # Start the person counter at zero for this frame
    person_count = 0

    # Check every detected object
    for box in result.boxes:

        # Get the object's class number
        class_id = int(box.cls[0])

        # Convert class number to class name
        class_name = model.names[class_id]

        # Get confidence score
        confidence = float(box.conf[0])

        # Count only people
        if class_name == "person":
            person_count += 1

    # Draw all YOLO detections
    annotated_frame = result.plot()

    # Display the number of people on the video
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
    cv2.imshow("People Counter", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()