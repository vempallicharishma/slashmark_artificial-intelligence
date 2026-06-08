import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Webcam input
cap = cv2.VideoCapture("objects_video.mp4")

# Uncomment for video file
# cap = cv2.VideoCapture("room.mp4")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    h, w, _ = frame.shape

    # Divide frame into 3 zones
    left_boundary = w // 3
    right_boundary = 2 * w // 3

    cv2.line(frame, (left_boundary, 0),
             (left_boundary, h), (255, 0, 0), 2)

    cv2.line(frame, (right_boundary, 0),
             (right_boundary, h), (255, 0, 0), 2)

    action = "MOVE FORWARD"
    largest_area = 0

    # Run YOLO detection
    results = model(frame, verbose=False)

    for result in results:

        boxes = result.boxes

        for box in boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            conf = float(box.conf[0])

            cls = int(box.cls[0])

            label = model.names[cls]

            area = (x2 - x1) * (y2 - y1)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Draw bounding box
            cv2.rectangle(frame,
                          (x1, y1),
                          (x2, y2),
                          (0, 255, 0), 2)

            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            cv2.putText(frame,
                        f"{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2)

            # Determine zone
            if cx < left_boundary:
                zone = "LEFT"

            elif cx > right_boundary:
                zone = "RIGHT"

            else:
                zone = "CENTER"

            if area > largest_area:
                largest_area = area

                # Navigation logic
                if area > 90000:
                    action = "STOP"

                elif zone == "CENTER" and area > 50000:
                    action = "TURN LEFT"

                elif zone == "LEFT":
                    action = "MOVE RIGHT"

                elif zone == "RIGHT":
                    action = "MOVE LEFT"

                else:
                    action = "MOVE FORWARD"

                cv2.putText(frame,
                            f"Zone: {zone}",
                            (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 0),
                            2)

    # Display action
    color = (0, 255, 0)

    if action == "STOP":
        color = (0, 0, 255)

    elif "TURN" in action:
        color = (0, 255, 255)

    cv2.putText(frame,
                f"ACTION: {action}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                3)

    cv2.imshow("AI Indoor Obstacle Avoidance", frame)

    key = cv2.waitKey(1)

    if key == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()

