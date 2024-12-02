import cv2
import numpy as np
import time
from collections import deque
import imutils

# Define the video path
video_path = "/Users/amanthaper/Downloads/Squares.mp4"

# Define HSV color range for red-orange
red_orange_lower = (5, 100, 100)
red_orange_upper = (20, 255, 255)

# Initialize parameters for square drill detection
movement_threshold = 50
square_pattern_threshold = 10
buffer_size = 64

# Initialize parameters for intensity and repetition counting
total_repetitions = 0
start_position = None
moved_out = False
return_threshold = 50

# Initialize tracking deque and other variables
pts = deque(maxlen=buffer_size)
drill_count = 0
current_drill_reps = 0
intensity = "Low"
last_time = time.time()

# Open the video file
cap = cv2.VideoCapture(video_path)

def calculate_distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and process the frame
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Create a mask for the red-orange color
    mask = cv2.inRange(hsv, red_orange_lower, red_orange_upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours in the mask
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    football_position = None

    # Only proceed if at least one contour is found
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        football_position = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # Draw the circle and centroid on the frame
        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 4)
            cv2.circle(frame, football_position, 5, (0, 0, 255), -1)

    # Update points deque
    pts.appendleft(football_position)

    # Detect if a square drill is completed
    if football_position:
        if start_position is None:
            start_position = football_position

        distance_from_start = calculate_distance(football_position, start_position)

        if distance_from_start > movement_threshold:
            moved_out = True

        # Count as a repetition if the ball returns to start position after moving out
        if moved_out and distance_from_start < return_threshold:
            total_repetitions += 1
            drill_count += 1
            current_drill_reps += 1
            start_position = football_position  # Reset start position
            moved_out = False  # Reset movement flag

            # Calculate intensity based on recent repetition speed
            if drill_count >= 2:
                time_diff = time.time() - last_time
                if time_diff < 0.4:
                    intensity = "High"
                elif time_diff < 0.6:
                    intensity = "Medium"
                else:
                    intensity = "Low"
                last_time = time.time()  # Reset time
                drill_count = 0  # Reset drill count

    # Display repetition, intensity, and movement information
    cv2.putText(frame, f"Total Reps: {total_repetitions}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Intensity: {intensity}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Show the frame
    cv2.imshow("Football Tracking and Intensity Display", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

