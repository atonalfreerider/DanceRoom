import cv2
import numpy as np
import json
from tqdm import tqdm


def load_yolo_detections(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def point_in_box(point, box):
    x, y = point
    x1, y1, x2, y2 = box
    return x1 < x < x2 and y1 < y < y2


def filter_static_points(points, old_points, threshold=0.1):
    if old_points is None:
        return points
    # Ensure we're comparing the same number of points
    min_points = min(len(points), len(old_points))
    points = points[:min_points]
    old_points = old_points[:min_points]
    distances = np.linalg.norm(points - old_points, axis=1)
    return points[distances > threshold]



def room_tracker(input_path, output_path, debug_output_path, yolo_detections_path):
    # Load YOLO detections
    yolo_detections = load_yolo_detections(yolo_detections_path)

    # Initialize video capture
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get video properties for debug output
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Set up debug video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    debug_out = cv2.VideoWriter(debug_output_path, fourcc, fps, (width, height))

    # Camera matrix (you may need to calibrate your camera to get accurate values)
    focal_length = 1000
    center = (width // 2, height // 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)

    # Distortion coefficients (assume no distortion)
    dist_coeffs = np.zeros((4, 1))

    # Initialize feature detector and tracker
    feature_params = dict(qualityLevel=0.01, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Read first frame and detect initial features
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # Create a mask to exclude logo area (adjust coordinates as needed)
    logo_mask = np.ones(old_gray.shape[:2], dtype=np.uint8)
    logo_mask[0:50, 0:50] = 0  # Assume logo is in top-left corner, adjust as needed

    p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=500, mask=logo_mask, **feature_params)

    # Initialize variables for storing deltas and tracks
    deltas = []
    prev_angles = None
    tracks = []
    max_track_length = int(5 * fps)  # 5 seconds of tracks

    frame_count = 0

    # Progress bar
    pbar = tqdm(total=total_frames, desc="Processing frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Get YOLO detections for the current frame
        current_detections = yolo_detections.get(str(frame_count), [])
        current_boxes = [detection["bbox"] for detection in current_detections]

        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # Filter out points inside YOLO detection boxes
            good_indices = [i for i, point in enumerate(good_new)
                            if not any(point_in_box(point, box) for box in current_boxes)]
            good_new = good_new[good_indices]
            good_old = good_old[good_indices]

            # Filter out static points
            good_new = filter_static_points(good_new, good_old)
        else:
            good_new = np.empty((0, 2))

        # If we don't have enough points, detect new ones
        while len(good_new) < 100:
            additional_points = cv2.goodFeaturesToTrack(frame_gray, mask=logo_mask,
                                                        maxCorners=500 - len(good_new), **feature_params)
            if additional_points is None:
                break
            additional_points = additional_points.reshape(-1, 2)

            # Filter out points inside YOLO detection boxes
            additional_points = [point for point in additional_points
                                 if not any(point_in_box(point, box) for box in current_boxes)]

            good_new = np.vstack((good_new, additional_points))

        # Ensure we have at least 20 points
        if len(good_new) > 20:
            good_new = good_new[:20]

        # Update tracks
        if len(tracks) < len(good_new):
            for _ in range(len(good_new) - len(tracks)):
                tracks.append([])

        for i, new in enumerate(good_new):
            if i < len(tracks):
                tracks[i].append((frame_count, tuple(new.ravel())))
                # Remove old points
                while tracks[i] and frame_count - tracks[i][0][0] >= max_track_length:
                    tracks[i].pop(0)

        # Estimate camera pose
        if len(good_new) >= 4:
            # Create 3D points assuming all points are on z=0 plane
            obj_points = np.hstack((good_new, np.zeros((good_new.shape[0], 1)))).astype(np.float32)

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

        # Clear the mask
        mask = np.zeros_like(old_frame)

        # Draw the tracks
        for track in tracks:
            if len(track) > 1:
                for i in range(1, len(track)):
                    pt1 = tuple(map(int, track[i - 1][1]))
                    pt2 = tuple(map(int, track[i][1]))
                    cv2.line(mask, pt1, pt2, (0, 255, 0), 2)

                # Draw the current point
                cv2.circle(frame, tuple(map(int, track[-1][1])), 5, (0, 0, 255), -1)

        img = cv2.add(frame, mask)

        # Draw YOLO detection boxes
        for box in current_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Add text with current angles
        if prev_angles:
            cv2.putText(img, f"Pan: {pan:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, f"Tilt: {tilt:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, f"Roll: {roll:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write the frame to debug video
        debug_out.write(img)

        # Update the previous frame and points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

        frame_count += 1
        pbar.update(1)

    cap.release()
    debug_out.release()
    pbar.close()

    # Save deltas to JSON file
    try:
        with open(output_path, 'w') as f:
            json.dump(deltas, f, indent=2)
        print(f"Successfully wrote deltas to {output_path}")
    except Exception as e:
        print(f"Error writing JSON file: {str(e)}")
        print("Returning deltas as a list instead.")

    return deltas

