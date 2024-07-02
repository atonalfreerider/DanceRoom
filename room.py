import cv2
import numpy as np
import os

# Initialize video capture
cap = cv2.VideoCapture('/home/john/Downloads/carlos-aline-spin-cam1.mp4')

# Initialize ORB detector
orb = cv2.ORB_create()

# Variables to store previous frame data
prev_kp = None
prev_des = None
prev_frame = None

# Create a matcher for feature matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Directory to save frames with grid overlay
output_dir = '/home/john/Desktop/out'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and descriptors
    kp, des = orb.detectAndCompute(gray, None)

    if prev_kp is not None:
        # Match features between previous and current frame
        matches = bf.match(prev_des, des)

        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract the matched keypoints
        prev_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches])
        curr_pts = np.float32([kp[m.trainIdx].pt for m in matches])

        # Find the homography matrix
        H, mask = cv2.findHomography(prev_pts, curr_pts, cv2.RANSAC, 5.0)

        # Decompose the homography matrix to get rotation and translation
        if H is not None:
            _, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, np.eye(3))
            # Assuming the first solution is the correct one
            R = Rs[0]
            T = Ts[0]
            N = Ns[0]

            # Calculate pan, tilt, and roll from rotation matrix
            pan = np.arctan2(R[1, 0], R[0, 0])
            tilt = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
            roll = np.arctan2(R[2, 1], R[2, 2])

            # Calculate zoom (assuming focal length is changing)
            zoom = np.linalg.norm(T)

            # Print pan, tilt, roll, and zoom
            print(
                f"Pan: {np.degrees(pan):.2f}, Tilt: {np.degrees(tilt):.2f}, Roll: {np.degrees(roll):.2f}, Zoom: {zoom:.2f}")

            # Draw grid lines on the floor plane
            grid_size = 50  # Size of the grid cells
            for i in range(0, frame.shape[1], grid_size):
                for j in range(0, frame.shape[0], grid_size):
                    start_point = (i, j)
                    end_point_x = (i + grid_size, j)
                    end_point_y = (i, j + grid_size)
                    frame = cv2.line(frame, start_point, end_point_x, (0, 255, 0), 1)
                    frame = cv2.line(frame, start_point, end_point_y, (0, 255, 0), 1)

    # Save the frame to the output directory
    frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
    cv2.imwrite(frame_filename, frame)

    # Update previous frame data
    prev_kp = kp
    prev_des = des
    prev_frame = frame
    frame_count += 1

# Release the video capture
cap.release()
cv2.destroyAllWindows()
