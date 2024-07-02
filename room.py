import cv2
import numpy as np
import os


def estimate_camera_params(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video")
        return None

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect corners
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
    corners = np.int32(corners)  # Changed from np.int0 to np.int32

    # Assume the four corners of the room are among the detected corners
    # For simplicity, we'll just use the extreme corners
    x_sorted = sorted(corners, key=lambda x: x[0][0])
    y_sorted = sorted(corners, key=lambda x: x[0][1])

    top_left = x_sorted[0][0]
    top_right = x_sorted[-1][0]
    bottom_left = y_sorted[-1][0]
    bottom_right = [x_sorted[-1][0][0], y_sorted[-1][0][1]]

    # Define 3D points of a rectangular room (assuming dimensions)
    room_width = 5.0  # meters
    room_length = 8.0  # meters
    object_points = np.array([
        (0, 0, 0),
        (room_width, 0, 0),
        (0, room_length, 0),
        (room_width, room_length, 0)
    ], dtype=np.float32)

    # Define corresponding 2D points
    image_points = np.array([
        top_left,
        top_right,
        bottom_left,
        bottom_right
    ], dtype=np.float32)

    # Camera matrix (assuming some values)
    focal_length = frame.shape[1]
    center = (frame.shape[1] / 2, frame.shape[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype=np.float32
    )

    # Assume no lens distortion
    dist_coeffs = np.zeros((4, 1))

    # Solve PnP
    success, rotation_vector, translation_vector = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Extract pan, tilt, roll
    pan = np.arctan2(rotation_matrix[0][2], rotation_matrix[2][2]) * 180 / np.pi
    tilt = np.arcsin(-rotation_matrix[1][2]) * 180 / np.pi
    roll = np.arctan2(rotation_matrix[1][0], rotation_matrix[1][1]) * 180 / np.pi

    # Estimate zoom (field of view)
    fov = 2 * np.arctan(frame.shape[1] / (2 * focal_length)) * 180 / np.pi

    cap.release()

    return {
        'pan': pan,
        'tilt': tilt,
        'roll': roll,
        'zoom': fov,
        'rotation_matrix': rotation_matrix,
        'translation_vector': translation_vector,
        'camera_matrix': camera_matrix
    }


def draw_floor_grid(frame, camera_params):
    # Define grid points on the floor
    x, y = np.meshgrid(range(0, 6), range(0, 9))
    z = np.zeros_like(x)
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # Project 3D points to 2D image plane
    image_points, _ = cv2.projectPoints(
        points.astype(np.float32),
        camera_params['rotation_matrix'],
        camera_params['translation_vector'],
        camera_params['camera_matrix'],
        None
    )

    # Reshape image_points for easier indexing
    image_points = image_points.reshape(-1, 2)

    # Draw grid lines
    for i in range(6):
        pt1 = tuple(map(int, image_points[i * 9]))
        pt2 = tuple(map(int, image_points[i * 9 + 8]))
        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    for i in range(9):
        pt1 = tuple(map(int, image_points[i]))
        pt2 = tuple(map(int, image_points[i + 45]))
        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    return frame


def process_video(video_path, output_folder):
    # Estimate camera parameters
    camera_params = estimate_camera_params(video_path)
    if camera_params is None:
        return

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file again
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw floor grid
        frame_with_grid = draw_floor_grid(frame.copy(), camera_params)

        # Save the frame
        output_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(output_path, frame_with_grid)

        frame_count += 1

    cap.release()

    print(f"Camera parameters:")
    print(f"Pan: {camera_params['pan']:.2f} degrees")
    print(f"Tilt: {camera_params['tilt']:.2f} degrees")
    print(f"Roll: {camera_params['roll']:.2f} degrees")
    print(f"Zoom (FOV): {camera_params['zoom']:.2f} degrees")
    print(f"Processed {frame_count} frames")


# Usage
video_path = "/home/john/Downloads/carlos-aline-spin-cam1.mp4"
output_folder = "/home/john/Desktop/out"
process_video(video_path, output_folder)