import cv2
import numpy as np
from collections import deque


def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges


def detect_lines(edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=50)
    return lines


def find_longest_lines(lines, num_lines=4):
    if lines is None:
        return []
    sorted_lines = sorted(lines, key=lambda x: np.sqrt((x[0][2] - x[0][0]) ** 2 + (x[0][3] - x[0][1]) ** 2),
                          reverse=True)
    return sorted_lines[:num_lines]


def classify_lines(lines, frame_height):
    floor_lines = []
    wall_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if y1 > frame_height / 2 and y2 > frame_height / 2:
            floor_lines.append(line)
        else:
            wall_lines.append(line)
    return floor_lines, wall_lines


def find_vanishing_points(lines):
    if len(lines) < 2:
        return None

    points = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            x1, y1, x2, y2 = lines[i][0]
            x3, y3, x4, y4 = lines[j][0]

            A = np.array([
                [y2 - y1, x1 - x2],
                [y4 - y3, x3 - x4]
            ])
            b = np.array([x1 * y2 - x2 * y1, x3 * y4 - x4 * y3])

            try:
                point = np.linalg.solve(A, b)
                points.append(point)
            except np.linalg.LinAlgError:
                continue

    if not points:
        return None

    return np.mean(points, axis=0)


def estimate_floor_plane(floor_lines, wall_lines, frame_shape):
    floor_vp = find_vanishing_points(floor_lines)
    wall_vp = find_vanishing_points(wall_lines)

    if floor_vp is None or wall_vp is None:
        return None

    # Estimate floor corners
    height, width = frame_shape[:2]
    corners = [
        [0, height],
        [width, height],
        [width, int(wall_vp[1])],
        [0, int(wall_vp[1])]
    ]

    return np.array(corners, dtype=np.float32)


def estimate_camera_pose(floor_plane, focal_length):
    # Simplified camera pose estimation
    center = np.mean(floor_plane, axis=0)
    width = np.linalg.norm(floor_plane[1] - floor_plane[0])
    height = np.linalg.norm(floor_plane[3] - floor_plane[0])

    pan = np.arctan2(center[0] - focal_length, focal_length)
    tilt = np.arctan2(center[1] - focal_length, focal_length)
    roll = np.arctan2(height, width) - np.pi / 2
    zoom = focal_length / (width * height) ** 0.25

    return np.array([pan, tilt, roll, zoom])


def smooth_poses(poses, window_size=5):
    smoothed = []
    for i in range(len(poses)):
        start = max(0, i - window_size + 1)
        window = poses[start:i + 1]
        smoothed.append(np.mean(window, axis=0))
    return np.array(smoothed)


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    floor_planes = []
    camera_poses = []

    focal_length = 1000  # Assume a focal length (adjust as needed)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        edges = preprocess_frame(frame)
        lines = detect_lines(edges)
        longest_lines = find_longest_lines(lines)

        floor_lines, wall_lines = classify_lines(longest_lines, frame.shape[0])

        floor_plane = estimate_floor_plane(floor_lines, wall_lines, frame.shape)
        if floor_plane is not None:
            floor_planes.append(floor_plane)

            camera_pose = estimate_camera_pose(floor_plane, focal_length)
            camera_poses.append(camera_pose)

    cap.release()

    # Find the most consistent floor plane
    if floor_planes:
        universal_floor_plane = np.mean(floor_planes, axis=0)
    else:
        universal_floor_plane = None

    # Smooth camera poses
    smoothed_poses = smooth_poses(camera_poses)

    return universal_floor_plane, smoothed_poses


# Usage
video_path = "/home/john/Downloads/carlos-aline-spin-cam2.mp4"
floor_plane, camera_poses = process_video(video_path)

print("Universal Floor Plane:")
print(floor_plane)
print("\nCamera Poses (Pan, Tilt, Roll, Zoom):")
print(camera_poses)