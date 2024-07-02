import cv2
import numpy as np
import random
import os
from scipy.cluster.vq import kmeans, vq


def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges


def cluster_lines(lines, n_clusters=4):
    if lines is None or len(lines) < n_clusters:
        return []

    points = np.array([(line[0][0], line[0][1], line[0][2], line[0][3]) for line in lines], dtype=np.float64)

    if len(points) < n_clusters:
        return [lines]  # Return all lines as a single cluster if there are fewer lines than clusters

    centroids, _ = kmeans(points, n_clusters)
    labels, _ = vq(points, centroids)

    clustered_lines = [[] for _ in range(n_clusters)]
    for i, label in enumerate(labels):
        clustered_lines[label].append(lines[i])

    return [cluster for cluster in clustered_lines if cluster]  # Remove empty clusters


def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges


def detect_lines(edges, frame_height, frame_width):
    # Separate detection for lower and upper half of the frame
    lower_half = edges[frame_height // 2:, :]
    upper_half = edges[:frame_height // 2, :]

    # Parameters for Hough Line Transform
    rho = 1
    theta = np.pi / 180
    threshold = 50
    min_line_length = 50
    max_line_gap = 10

    # Detect lines in lower half (emphasize horizontal and diagonal)
    lower_lines = cv2.HoughLinesP(lower_half, rho, theta, threshold,
                                  minLineLength=min_line_length,
                                  maxLineGap=max_line_gap)

    # Detect lines in upper half (emphasize vertical)
    upper_lines = cv2.HoughLinesP(upper_half, rho, theta, threshold,
                                  minLineLength=min_line_length,
                                  maxLineGap=max_line_gap)

    # Adjust y-coordinates for upper lines
    if upper_lines is not None:
        upper_lines[:, :, 1] += frame_height // 2
        upper_lines[:, :, 3] += frame_height // 2

    # Combine lower and upper lines
    all_lines = np.vstack((lower_lines,
                           upper_lines)) if upper_lines is not None and lower_lines is not None else lower_lines or upper_lines

    return all_lines


def classify_lines(lines, frame_height, frame_width):
    floor_lines = []
    wall_lines = []
    if lines is None:
        return floor_lines, wall_lines

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        angle = np.abs(np.arctan2(dy, dx) * 180 / np.pi)

        # Classify based on angle and position
        if y1 > frame_height * 0.6 and y2 > frame_height * 0.6:  # Lower 40% of the frame
            if angle < 30 or angle > 150:  # More horizontal
                floor_lines.append(line)
        elif y1 < frame_height * 0.4 and y2 < frame_height * 0.4:  # Upper 40% of the frame
            if 60 < angle < 120:  # More vertical
                wall_lines.append(line)

    return floor_lines, wall_lines


def draw_lines(frame, lines, color):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), color, 2)
    return frame


def ransac_vanishing_point(lines, num_iterations=100, threshold=10):
    best_vp = None
    max_inliers = 0

    for _ in range(num_iterations):
        if len(lines) < 2:
            continue

        # Randomly select two lines
        line1, line2 = random.sample(lines, 2)

        x1, y1, x2, y2 = line1[0]
        x3, y3, x4, y4 = line2[0]

        A = np.array([
            [y2 - y1, x1 - x2],
            [y4 - y3, x3 - x4]
        ])
        b = np.array([x1 * y2 - x2 * y1, x3 * y4 - x4 * y3])

        try:
            vp = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue

        # Count inliers
        inliers = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            d = abs((y2 - y1) * vp[0] - (x2 - x1) * vp[1] + x2 * y1 - y2 * x1) / np.sqrt(
                (y2 - y1) ** 2 + (x2 - x1) ** 2)
            if d < threshold:
                inliers += 1

        if inliers > max_inliers:
            max_inliers = inliers
            best_vp = vp

    return best_vp


def estimate_floor_plane(floor_lines, wall_lines, frame_shape):
    floor_vp = ransac_vanishing_point(floor_lines)
    wall_vp = ransac_vanishing_point(wall_lines)

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


def pnp_camera_pose(floor_plane, camera_matrix):
    # Assume world coordinates of floor plane
    world_points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0]
    ], dtype=np.float32)

    success, rotation_vec, translation_vec = cv2.solvePnP(world_points, floor_plane, camera_matrix, None)

    if not success:
        return None

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)

    # Extract pan, tilt, roll from rotation matrix
    pan = np.arctan2(rotation_mat[0, 2], rotation_mat[2, 2])
    tilt = np.arcsin(-rotation_mat[1, 2])
    roll = np.arctan2(rotation_mat[1, 0], rotation_mat[1, 1])

    # Calculate zoom based on translation
    zoom = 1 / max(translation_vec[2], 1e-6)  # Avoid division by zero

    # Ensure all values are scalar
    pan = float(pan)
    tilt = float(tilt)
    roll = float(roll)
    zoom = float(zoom)

    return np.array([pan, tilt, roll, zoom], dtype=np.float64)


def smooth_poses(poses, window_size=5):
    smoothed = []
    for i in range(len(poses)):
        start = max(0, i - window_size + 1)
        window = poses[start:i + 1]
        smoothed.append(np.mean(window, axis=0))
    return np.array(smoothed)


def is_keyframe(current_pose, previous_pose, threshold=0.1):
    if previous_pose is None:
        return True
    return np.any(np.abs(current_pose - previous_pose) > threshold)


def visualize_frame(frame, floor_plane, camera_pose):
    # Draw floor plane
    if floor_plane is not None:
        cv2.polylines(frame, [floor_plane.astype(int)], True, (0, 255, 0), 2)

    # Display camera pose information
    if camera_pose is not None:
        pan, tilt, roll, zoom = camera_pose
        cv2.putText(frame, f"Pan: {pan:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Tilt: {tilt:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Roll: {roll:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Zoom: {zoom:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

def process_video(video_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)

    # Estimate camera matrix
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    focal_length = frame_width
    camera_matrix = np.array([
        [focal_length, 0, frame_width / 2],
        [0, focal_length, frame_height / 2],
        [0, 0, 1]
    ], dtype=np.float32)

    floor_planes = []
    camera_poses = []

    previous_pose = None
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        edges = preprocess_frame(frame)
        lines = detect_lines(edges, frame.shape[0], frame.shape[1])

        floor_lines, wall_lines = classify_lines(lines, frame.shape[0], frame.shape[1])

        # Draw lines
        frame_with_lines = frame.copy()
        frame_with_lines = draw_lines(frame_with_lines, floor_lines, (0, 255, 0))  # Green for floor
        frame_with_lines = draw_lines(frame_with_lines, wall_lines, (0, 0, 255))  # Red for walls

        # Save frame with lines
        cv2.imwrite(os.path.join(output_dir, f"frame_{frame_count:04d}_lines.jpg"), frame_with_lines)

        floor_plane = estimate_floor_plane(floor_lines, wall_lines, frame.shape)
        if floor_plane is not None:
            camera_pose = pnp_camera_pose(floor_plane, camera_matrix)

            if camera_pose is not None and is_keyframe(camera_pose, previous_pose):
                floor_planes.append(floor_plane)
                camera_poses.append(camera_pose)
                previous_pose = camera_pose

                # Visualize the frame
                visualized_frame = visualize_frame(frame.copy(), floor_plane, camera_pose)
                cv2.imwrite(os.path.join(output_dir, f"frame_{frame_count:04d}_visualized.jpg"), visualized_frame)

        frame_count += 1

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
output_dir = "/home/john/Desktop/out"
floor_plane, camera_poses = process_video(video_path, output_dir)

print("Universal Floor Plane:")
print(floor_plane)
print("\nCamera Poses (Pan, Tilt, Roll, Zoom):")
print(camera_poses)
