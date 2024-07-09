import cv2
import numpy as np
import json
from tqdm import tqdm


def room_tracker(input_path, output_path):
    # Initialize video capture
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Camera matrix (you may need to calibrate your camera to get accurate values)
    focal_length = 1000
    center = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2),
              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2))
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)

    # Distortion coefficients (assume no distortion)
    dist_coeffs = np.zeros((4, 1))

    # Initialize feature detector and tracker
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Read first frame and detect initial features
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Initialize variables for storing deltas
    deltas = []
    prev_angles = None

    # Progress bar
    pbar = tqdm(total=total_frames, desc="Processing frames")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

        # Estimate camera pose
        if len(good_new) >= 4:
            # Create 3D points assuming all points are on z=0 plane
            obj_points = np.hstack((good_old, np.zeros((good_old.shape[0], 1)))).astype(np.float32)

            # Use PnP to estimate camera pose
            success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
                obj_points, good_new, camera_matrix, dist_coeffs
            )

            if success:
                # Convert rotation vector to Euler angles
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                euler_angles = cv2.decomposeProjectionMatrix(np.hstack((rotation_matrix, translation_vector)))[6]

                pan, tilt, roll = [angle[0] for angle in euler_angles]

                # Calculate deltas
                if prev_angles is not None:
                    delta_pan = pan - prev_angles[0]
                    delta_tilt = tilt - prev_angles[1]
                    delta_roll = roll - prev_angles[2]
                    deltas.append({
                        "frame": frame_count,
                        "delta_pan": float(delta_pan),
                        "delta_tilt": float(delta_tilt),
                        "delta_roll": float(delta_roll)
                    })

                prev_angles = (pan, tilt, roll)

        # Update the previous frame and points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

        frame_count += 1
        pbar.update(1)

    cap.release()
    pbar.close()

    # Save deltas to JSON file
    with open(output_path, 'w') as f:
        json.dump(deltas, f, indent=2)

    return deltas


def debug_render(input_path, deltas, output_path):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Create a grid of points
    grid_size = 10
    x = np.linspace(0, width, grid_size)
    y = np.linspace(0, height, grid_size)
    points = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)

    # Initialize cumulative angles
    cum_pan, cum_tilt, cum_roll = 0, 0, 0

    pbar = tqdm(total=len(deltas), desc="Rendering debug video")

    for i, delta in enumerate(deltas):
        ret, frame = cap.read()
        if not ret:
            break

        cum_pan += delta['delta_pan']
        cum_tilt += delta['delta_tilt']
        cum_roll += delta['delta_roll']

        # Create rotation matrix
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(cum_tilt), -np.sin(cum_tilt)],
                        [0, np.sin(cum_tilt), np.cos(cum_tilt)]])
        R_y = np.array([[np.cos(cum_pan), 0, np.sin(cum_pan)],
                        [0, 1, 0],
                        [-np.sin(cum_pan), 0, np.cos(cum_pan)]])
        R_z = np.array([[np.cos(cum_roll), -np.sin(cum_roll), 0],
                        [np.sin(cum_roll), np.cos(cum_roll), 0],
                        [0, 0, 1]])
        R = R_z @ R_y @ R_x

        # Project points
        points_3d = np.hstack((points, np.zeros((points.shape[0], 1))))
        projected_points = (R @ points_3d.T).T[:, :2]

        # Draw grid
        for point in projected_points:
            cv2.circle(frame, tuple(point.astype(int)), 3, (0, 255, 0), -1)

        # Draw lines
        for i in range(grid_size):
            start = i * grid_size
            end = start + grid_size
            cv2.polylines(frame, [projected_points[start:end].astype(np.int32)], False, (0, 255, 0), 1)
            cv2.polylines(frame, [projected_points[i::grid_size].astype(np.int32)], False, (0, 255, 0), 1)

        # Add text with current angles
        cv2.putText(frame, f"Pan: {cum_pan:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Tilt: {cum_tilt:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Roll: {cum_roll:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(frame)
        pbar.update(1)

    cap.release()
    out.release()
    pbar.close()
