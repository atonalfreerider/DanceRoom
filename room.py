import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import random


def extract_frames(video_path, interval):
    frames = []
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames


def detect_floor_plane(frame):
    height, width = frame.shape[:2]
    lower_half = frame[height // 2:, :]

    gray = cv2.cvtColor(lower_half, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    if lines is None:
        return None

    horizontal_lines = []
    for line in lines:
        rho, theta = line[0]
        if abs(theta - np.pi / 2) < np.pi / 18:  # Within 10 degrees of horizontal
            horizontal_lines.append((rho, theta))

    if len(horizontal_lines) < 2:
        return None

    # Use RANSAC to fit a plane (in this case, just a line) to the horizontal lines
    best_model = None
    best_inliers = 0
    for _ in range(100):  # RANSAC iterations
        sample = random.sample(horizontal_lines, 2)
        try:
            model = np.polyfit([s[0] for s in sample], [s[1] for s in sample], 1)
            inliers = sum(abs(model[0] * line[0] + model[1] - line[1]) < 0.1 for line in horizontal_lines)
            if inliers > best_inliers:
                best_model = model
                best_inliers = inliers
        except np.RankWarning:
            continue  # Skip this iteration if polyfit is poorly conditioned

    return best_model

def estimate_camera_pose(floor_model, estimated_intrinsics):
    if floor_model is None:
        return None

    # Assuming the floor plane is y = ax + b in the image
    a, b = floor_model

    # Convert to 3D plane normal
    normal = np.array([a, -1, 0])
    normal /= np.linalg.norm(normal)

    # Assuming the camera is looking down at the floor, we can estimate its orientation
    camera_up = np.array([0, 1, 0])
    camera_forward = np.cross(normal, camera_up)
    camera_right = np.cross(camera_forward, camera_up)

    rotation_matrix = np.column_stack((camera_right, camera_up, camera_forward))

    # Convert rotation matrix to quaternion
    r = R.from_matrix(rotation_matrix)
    quaternion = r.as_quat()

    # Estimate translation (this is a simplification and may need refinement)
    translation = np.array([0, b, 0])

    return (quaternion, translation)


def track_camera_motion(prev_frame, curr_frame, prev_pose):
    if prev_pose is None:
        return None

    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Calculate the average flow vector
    avg_flow = np.mean(flow, axis=(0, 1))

    # Update the camera pose based on the average flow
    prev_quat, prev_trans = prev_pose

    # Simplified update (this needs refinement for accurate results)
    updated_trans = prev_trans + np.array([avg_flow[0], 0, avg_flow[1]])

    return (prev_quat, updated_trans)


def detect_zoom_change(prev_frame, curr_frame):
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Detect features in both frames
    orb = cv2.ORB_create()
    prev_kp, prev_des = orb.detectAndCompute(prev_gray, None)
    curr_kp, curr_des = orb.detectAndCompute(curr_gray, None)

    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(prev_des, curr_des)

    # Calculate average distance between matched features
    avg_dist = np.mean([m.distance for m in matches])

    # If average distance decreased, zoom in; if increased, zoom out
    zoom_factor = 1.0
    if avg_dist < 32:  # Threshold may need adjustment
        zoom_factor = 32 / avg_dist
    elif avg_dist > 32:
        zoom_factor = avg_dist / 32

    return zoom_factor


def draw_floor_grid(frame, floor_model, camera_pose, intrinsics):
    height, width = frame.shape[:2]

    if floor_model is None or camera_pose is None:
        return frame

    # Create a grid in 3D space
    grid_size = 10
    grid_step = 1
    x, z = np.meshgrid(range(-grid_size, grid_size + 1, grid_step), range(-grid_size, grid_size + 1, grid_step))
    y = np.zeros_like(x)

    # Create homogeneous coordinates
    points_3d = np.vstack((x.ravel(), y.ravel(), z.ravel(), np.ones_like(x.ravel())))

    # Camera extrinsics
    quat, trans = camera_pose
    rotation_matrix = R.from_quat(quat).as_matrix()
    extrinsics = np.hstack((rotation_matrix, trans.reshape(3, 1)))

    # Project 3D points to 2D
    points_cam = extrinsics @ points_3d
    points_2d = intrinsics @ points_cam[:3, :]

    # Handle division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        points_2d = points_2d[:2] / points_2d[2]
    points_2d = np.where(np.isfinite(points_2d), points_2d, 0)

    points_2d = points_2d.T.reshape(x.shape + (2,))

    # Draw the grid
    for i in range(points_2d.shape[0]):
        pts = points_2d[i].astype(np.int32)
        pts = pts[np.all(pts >= 0, axis=1) & (pts[:, 0] < width) & (pts[:, 1] < height)]
        if len(pts) > 1:
            cv2.polylines(frame, [pts], False, (0, 255, 0), 1)

    for i in range(points_2d.shape[1]):
        pts = points_2d[:, i].astype(np.int32)
        pts = pts[np.all(pts >= 0, axis=1) & (pts[:, 0] < width) & (pts[:, 1] < height)]
        if len(pts) > 1:
            cv2.polylines(frame, [pts], False, (0, 255, 0), 1)

    return frame


def main():
    video_path = "/home/john/Downloads/larissa-kadu-counter-demo/01-Demo-Ali.mp4"
    output_dir = "/home/john/Desktop/out"
    os.makedirs(output_dir, exist_ok=True)

    frames = extract_frames(video_path, interval=30)

    height, width = frames[0].shape[:2]
    estimated_intrinsics = np.array([[1000, 0, width / 2],
                                     [0, 1000, height / 2],
                                     [0, 0, 1]])

    camera_poses = []
    prev_pose = None

    for i, frame in enumerate(frames):
        floor_model = detect_floor_plane(frame)
        camera_pose = estimate_camera_pose(floor_model, estimated_intrinsics)

        if i > 0:
            tracked_pose = track_camera_motion(frames[i - 1], frame, prev_pose)
            if tracked_pose is not None:
                camera_pose = tracked_pose

            zoom_factor = detect_zoom_change(frames[i - 1], frame)
            estimated_intrinsics[0, 0] *= zoom_factor
            estimated_intrinsics[1, 1] *= zoom_factor

        camera_poses.append(camera_pose)
        prev_pose = camera_pose

        # Draw floor grid and save frame
        if camera_pose is not None:
            frame_with_grid = draw_floor_grid(frame.copy(), floor_model, camera_pose, estimated_intrinsics)
            quat, trans = camera_pose
            cv2.putText(frame_with_grid, f"Rotation: {quat}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_with_grid, f"Translation: {trans}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                        2)
            cv2.imwrite(os.path.join(output_dir, f"frame_{i:04d}.jpg"), frame_with_grid)
        else:
            cv2.imwrite(os.path.join(output_dir, f"frame_{i:04d}.jpg"), frame)

        print(f"Processed frame {i + 1}/{len(frames)}")


if __name__ == "__main__":
    main()