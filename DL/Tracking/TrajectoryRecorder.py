import cv2
import time
import csv
import numpy as np

from PingPong_detection import preprocess, read_Trackbar, creat_Trackbar

SAVE_PATH = "Trajectory_data/trajectory_data.csv"

cap = cv2.VideoCapture(0)
cv2.namedWindow('Frame', 2)
cv2.namedWindow('Trackbar', 2)
creat_Trackbar()

trajectory = []

print("[INFO] Recording trajectory. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    lower_yellow, upper_yellow = read_Trackbar()
    mask = preprocess(frame, lower_yellow, upper_yellow)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 10:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            x, y = int(x), int(y)
            timestamp = time.time()
            trajectory.append((timestamp, x, y))
            
            # draw
            cv2.circle(frame, (x, y), int(radius), (0, 255, 0), 2)
            cv2.putText(frame, f"({x},{y})", (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            break  # only track largest one

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save to CSV
print(f"[INFO] Saving {len(trajectory)} trajectory points to {SAVE_PATH}")
with open(SAVE_PATH, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "x", "y"])
    writer.writerows(trajectory)

cap.release()
cv2.destroyAllWindows()