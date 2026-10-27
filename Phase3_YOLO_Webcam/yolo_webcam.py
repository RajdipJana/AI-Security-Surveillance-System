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

    # Run YOLO with a 50% confidence threshold
    results = model(frame, conf=0.5)

    # Get the first result from the current frame
    result = results[0]

    # Go through every object detected in this frame
    for box in result.boxes:

        # Get the detected object's class number
        class_id = int(box.cls[0])

        # Convert the class number into its name
        class_name = model.names[class_id]

        # Get the confidence score
        confidence = float(box.conf[0])

        # Print the detection in the terminal
        print(f"Detected: {class_name} | Confidence: {confidence:.2f}")

    # Draw the detections on the webcam frame
    annotated_frame = result.plot()

    # Display the frame
    cv2.imshow("YOLO Webcam Detection", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()