import os
import cv2
import numpy as np
import json
from tqdm import tqdm
from scipy.signal import savgol_filter


def estimate_transform(src_points, dst_points):
    transform_matrix, inliers = cv2.estimateAffinePartial2D(src_points, dst_points, method=cv2.RANSAC,
                                                            ransacReprojThreshold=3.0)
    if transform_matrix is not None:
        dx, dy = transform_matrix[:2, 2]
        rotation = np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])
        scale = np.sqrt(transform_matrix[0, 0] ** 2 + transform_matrix[1, 0] ** 2)
        return dx, dy, rotation, scale, inliers
    return 0, 0, 0, 1, None


def smooth_values(values, window_size=15, poly_order=3):
    return savgol_filter(values, window_size, poly_order)


def flatten_zoom(zoom_values, threshold=0.001):
    flattened = np.zeros_like(zoom_values)
    cumulative_zoom = 1.0
    for i, zoom in enumerate(zoom_values):
        if abs(zoom - 1) > threshold:
            flattened[i] = zoom
            cumulative_zoom *= zoom
        else:
            flattened[i] = 1.0
    return flattened, cumulative_zoom


def room_tracker(input_path, output_dir):
    # if the deltas.json file already exists, skip this step
    if os.path.exists(output_dir + "/deltas.json"):
        print("Deltas file already exists. Skipping room tracking.")
        return

    yolo_detections_path = output_dir + "/detections.json"
    yolo_detections = load_json(yolo_detections_path)
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    feature_params = dict(qualityLevel=0.01, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    tracks = {}
    next_track_id = 0
    max_track_length = int(5 * fps)
    deltas = []

    pbar = tqdm(total=total_frames, desc="Processing frames")

    for frame_count in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_boxes = [detection["bbox"] for detection in yolo_detections.get(str(frame_count), [])]

        good_new = []
        good_old = []

        # Calculate optical flow for existing tracks
        if tracks:
            prev_points = np.float32([track['points'][-1][1] for track in tracks.values()]).reshape(-1, 1, 2)

            # optical flow takes the points from last frame and finds them in this frame
            curr_points, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, prev_points, None, **lk_params)

            good_new, good_old = [], []
            updated_tracks = {}

            for (track_id, track), (new, status) in zip(list(tracks.items()), zip(curr_points, st.ravel())):
                if status and not any(point_in_box(new[0], box) for box in current_boxes):
                    # only track points that are not inside of bounding boxes
                    track['points'].append((frame_count, tuple(new.ravel())))
                    track['last_seen'] = frame_count
                    while track['points'] and frame_count - track['points'][0][0] >= max_track_length:
                        track['points'].pop(0)
                    updated_tracks[track_id] = track
                    good_new.append(new.reshape(2))
                    good_old.append(prev_points[list(tracks.keys()).index(track_id)].reshape(2))

            tracks = updated_tracks

            # Filter out static points
            good_new, good_old = filter_static_points(good_new, good_old)

        if len(tracks) < 20:
            # top up points to 20, including points tracked on people
            mask = np.ones(frame_gray.shape[:2], dtype=np.uint8)
            for track in tracks.values():
                x, y = map(int, track['points'][-1][1])
                cv2.circle(mask, (x, y), 10, 0, -1)
            for box in current_boxes:
                # mask out points in people bounding boxes
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)

            new_points = cv2.goodFeaturesToTrack(frame_gray, mask=mask, maxCorners=20 - len(tracks), **feature_params)
            if new_points is not None:
                for point in new_points:
                    tracks[next_track_id] = {'points': [(frame_count, tuple(point.ravel()))], 'last_seen': frame_count}
                    next_track_id += 1

        good_new, good_old = np.array(good_new), np.array(good_old)

        delta_added = False
        if len(good_new) >= 4 and len(good_old) >= 4:
            # require 4 consistent points to do frame transform tracking
            dx, dy, rotation, scale, inliers = estimate_transform(good_old, good_new)

            if inliers is not None:
                delta_zoom = scale - 1

                deltas.append({
                    "frame": frame_count,
                    "delta_x": float(dx),
                    "delta_y": float(dy),
                    "delta_roll": float(rotation),
                    "delta_zoom": float(delta_zoom)
                })
                delta_added = True

        if not delta_added:
            deltas.append({
                "frame": frame_count,
                "delta_x": 0.0,
                "delta_y": 0.0,
                "delta_roll": 0.0,
                "delta_zoom": 0.0
            })

        old_gray = frame_gray.copy()
        pbar.update(1)

    cap.release()
    pbar.close()

    # Smooth and process the collected data
    rolls = smooth_values([d['delta_roll'] for d in deltas])
    zooms, cumulative_zoom = flatten_zoom([d['delta_zoom'] + 1 for d in deltas])

    # Update the deltas with smoothed values
    for i, delta in enumerate(deltas):
        delta['delta_roll'] = float(rolls[i])
        delta['delta_zoom'] = float(zooms[i] - 1)

    output_path = output_dir + "/deltas.json"
    with open(output_path, 'w') as f:
        json.dump(deltas, f, indent=2)
    print(f"Successfully wrote deltas to {output_path}")


def debug_video(input_path, output_dir, deltas_path):
    debug_output_path = output_dir + "/debug-points.mp4"
    if os.path.exists(debug_output_path):
        print("Debug video already exists. Skipping debug video creation.")
        return

    deltas = load_json(deltas_path)

    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)


    debug_out = cv2.VideoWriter(debug_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    crosshair_x = float(width // 2)
    crosshair_y = float(height // 2)
    crosshair_size = 20.0
    crosshair_roll = 0.0

    pbar = tqdm(total=total_frames, desc="Rendering crosshair frames")

    for frame_count in range(total_frames):
        ret, frame = cap.read()
        if not ret or frame_count >= len(deltas):
            break

        # Get the delta for this frame
        delta = deltas[frame_count]
        dx, dy = delta['delta_x'], delta['delta_y']
        rotation = delta['delta_roll']
        scale = delta['delta_zoom'] + 1

        crosshair_x += float(dx)
        crosshair_y += float(dy)
        crosshair_size *= scale
        crosshair_roll += rotation

        if crosshair_x < 0 or crosshair_x >= width or crosshair_y < 0 or crosshair_y >= height:
            # Reset crosshair if it goes out of bounds
            crosshair_x, crosshair_y = float(width // 2), float(height // 2)

        half_size = int(crosshair_size // 2)
        center = (int(crosshair_x), int(crosshair_y))
        rotation_matrix = cv2.getRotationMatrix2D(center, np.degrees(crosshair_roll), 1)

        p1 = np.array([center[0] - half_size, center[1], 1])
        p2 = np.array([center[0] + half_size, center[1], 1])
        p3 = np.array([center[0], center[1] - half_size, 1])
        p4 = np.array([center[0], center[1] + half_size, 1])

        p1_rotated = rotation_matrix.dot(p1)
        p2_rotated = rotation_matrix.dot(p2)
        p3_rotated = rotation_matrix.dot(p3)
        p4_rotated = rotation_matrix.dot(p4)

        img = frame.copy()
        cv2.line(img, tuple(p1_rotated[:2].astype(int)), tuple(p2_rotated[:2].astype(int)), (0, 255, 0), 2)
        cv2.line(img, tuple(p3_rotated[:2].astype(int)), tuple(p4_rotated[:2].astype(int)), (0, 255, 0), 2)

        debug_out.write(img)

        pbar.update(1)

    cap.release()
    pbar.close()
    debug_out.release()
    print(f"Successfully wrote debug video to {debug_output_path}")


def load_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)


def point_in_box(point, box):
    x, y = point
    x1, y1, x2, y2 = box
    return x1 < x < x2 and y1 < y < y2


def filter_static_points(points, old_points, threshold=0.1):
    if old_points is None or len(points) != len(old_points) or len(points) == 0 or len(old_points) == 0:
        return points, old_points

    # Convert lists to numpy arrays if they're not already
    points = np.array(points)
    old_points = np.array(old_points)

    # Calculate distances between points and old_points
    distances = np.linalg.norm(points - old_points, axis=1)

    # Create a mask for non-static points
    non_static_mask = distances > threshold

    # Return filtered points and old_points
    return points[non_static_mask], old_points[non_static_mask]
