import cv2
import numpy as np

# ---------------------------
# Region of Interest Function
# ---------------------------
def region_of_interest(image):
    height = image.shape[0]

    polygons = np.array([
        [
            (100, height),
            (image.shape[1]-100, height),
            (image.shape[1]//2, int(height*0.6))
        ]
    ])

    mask = np.zeros_like(image)

    cv2.fillPoly(mask, polygons, 255)

    masked_image = cv2.bitwise_and(image, mask)

    return masked_image


# ---------------------------
# Draw Lines
# ---------------------------
def draw_lines(image, lines):

    line_image = np.zeros_like(image)

    if lines is not None:

        for line in lines:

            x1, y1, x2, y2 = line.reshape(4)

            cv2.line(
                line_image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                5
            )

    return line_image


# ---------------------------
# Main Program
# ---------------------------

video = cv2.VideoCapture("road_video.mp4")

while video.isOpened():

    ret, frame = video.read()

    if not ret:
        break

    # Resize frame
    frame = cv2.resize(frame, (960, 540))

    # Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge Detection
    edges = cv2.Canny(blur, 50, 150)

    # Region of Interest
    cropped = region_of_interest(edges)

    # Hough Line Transform
    lines = cv2.HoughLinesP(
        cropped,
        2,
        np.pi / 180,
        100,
        np.array([]),
        minLineLength=40,
        maxLineGap=5
    )

    # Draw lane lines
    line_image = draw_lines(frame, lines)

    # Combine with original frame
    combo = cv2.addWeighted(
        frame,
        0.8,
        line_image,
        1,
        1
    )

    # Steering Logic
    direction = "STRAIGHT"

    if lines is not None:

        x_positions = []

        for line in lines:

            x1, y1, x2, y2 = line.reshape(4)

            x_positions.extend([x1, x2])

        lane_center = int(np.mean(x_positions))

        frame_center = frame.shape[1] // 2

        error = lane_center - frame_center

        # Draw centers
        cv2.circle(
            combo,
            (lane_center, frame.shape[0]//2),
            8,
            (255, 0, 0),
            -1
        )

        cv2.circle(
            combo,
            (frame_center, frame.shape[0]//2),
            8,
            (0, 0, 255),
            -1
        )

        if error < -50:
            direction = "LEFT"

        elif error > 50:
            direction = "RIGHT"

        else:
            direction = "STRAIGHT"

    # Display steering decision
    cv2.putText(
        combo,
        f"Direction: {direction}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.imshow("AI Self Driving Car", combo)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()