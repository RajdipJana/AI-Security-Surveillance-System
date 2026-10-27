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
    results = model.track(frame, conf=0.5, persist=True)

    # Get the first result
    result = results[0]

    # Draw bounding boxes and tracking IDs
    annotated_frame = result.plot()

    # Show the tracked video
    cv2.imshow("YOLO Person Tracking", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()