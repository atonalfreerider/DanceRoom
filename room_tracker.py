import cv2
import numpy as np

def room_tracker():
    # Initialize video capture (use 0 for default camera, or provide a video file path)
    cap = cv2.VideoCapture(0)

    # Define 3D points of a virtual cube in the room
    object_points = np.array([
        [0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
        [0, 0, -1], [0, 1, -1], [1, 1, -1], [1, 0, -1]
    ], dtype=np.float32)

    # Camera matrix (you may need to calibrate your camera to get accurate values)
    focal_length = 1000
    center = (320, 240)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)

    # Distortion coefficients (assume no distortion)
    dist_coeffs = np.zeros((4, 1))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect features in the frame
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        if len(keypoints) >= 4:
            # Convert keypoints to image points
            image_points = np.array([kp.pt for kp in keypoints[:8]], dtype=np.float32)

            # Solve PnP problem
            success, rotation_vector, translation_vector = cv2.solvePnPRansac(
                object_points, image_points, camera_matrix, dist_coeffs
            )

            if success:
                # Convert rotation vector to Euler angles
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                euler_angles = cv2.decomposeProjectionMatrix(np.hstack((rotation_matrix, translation_vector)))[6]

                pan, tilt, roll = [angle[0] for angle in euler_angles]

                # Display the results
                cv2.putText(frame, f"Pan: {pan:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Tilt: {tilt:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Roll: {roll:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Room Tracker', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    room_tracker()