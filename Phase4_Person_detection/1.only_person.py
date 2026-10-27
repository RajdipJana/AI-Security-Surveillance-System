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

    # Run YOLO on the frame
    results = model(frame, conf=0.5)

    # Get the first result
    result = results[0]

    # Check every detected object
    for box in result.boxes:

        # Get the class number
        class_id = int(box.cls[0])

        # Get the class name
        class_name = model.names[class_id]

        # Get confidence
        confidence = float(box.conf[0])

        # Only process people
        if class_name == "person":
            print(f"PERSON DETECTED | Confidence: {confidence:.2f}")

    # Draw all detections on the frame
    annotated_frame = result.plot()

    # Show the frame
    cv2.imshow("Person Detection", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release webcam
cap.release()

# Close windows
cv2.destroyAllWindows()