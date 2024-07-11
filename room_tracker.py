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
    if old_points is None or len(points) != len(old_points):
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


def estimate_transform(src_points, dst_points):
    transform_matrix, inliers = cv2.estimateAffinePartial2D(src_points, dst_points, method=cv2.RANSAC,
                                                            ransacReprojThreshold=3.0)
    if transform_matrix is not None:
        dx, dy = transform_matrix[:2, 2]
        rotation = np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])
        scale = np.sqrt(transform_matrix[0, 0] ** 2 + transform_matrix[1, 0] ** 2)
        return dx, dy, rotation, scale, inliers
    return 0, 0, 0, 1, None


def room_tracker(input_path, output_path, debug_output_path, yolo_detections_path):
    yolo_detections = load_yolo_detections(yolo_detections_path)
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    debug_out = cv2.VideoWriter(debug_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    feature_params = dict(qualityLevel=0.01, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    tracks = {}
    next_track_id = 0
    max_track_length = int(5 * fps)
    deltas = []

    crosshair_x = float(width // 2)
    crosshair_y = float(height // 2)
    initial_crosshair_size = 20
    crosshair_size = float(initial_crosshair_size)

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
            curr_points, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, prev_points, None, **lk_params)

            good_new, good_old = [], []
            updated_tracks = {}

            for (track_id, track), (new, status) in zip(list(tracks.items()), zip(curr_points, st.ravel())):
                if status and not any(point_in_box(new[0], box) for box in current_boxes):
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
            mask = np.ones(frame_gray.shape[:2], dtype=np.uint8)
            for track in tracks.values():
                x, y = map(int, track['points'][-1][1])
                cv2.circle(mask, (x, y), 10, 0, -1)
            for box in current_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)

            new_points = cv2.goodFeaturesToTrack(frame_gray, mask=mask, maxCorners=20 - len(tracks), **feature_params)
            if new_points is not None:
                for point in new_points:
                    tracks[next_track_id] = {'points': [(frame_count, tuple(point.ravel()))], 'last_seen': frame_count}
                    next_track_id += 1

        good_new, good_old = np.array(good_new), np.array(good_old)

        if len(good_new) >= 4 and len(good_old) >= 4:
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

                crosshair_x += float(dx)  # Move in opposite direction of camera motion
                crosshair_y += float(dy)  # Move in opposite direction of camera motion
                crosshair_size *= scale

                if crosshair_x < 0 or crosshair_x >= width or crosshair_y < 0 or crosshair_y >= height:
                    crosshair_x, crosshair_y = float(width // 2), float(height // 2)
                    crosshair_size = float(initial_crosshair_size)

        mask = np.zeros_like(frame)
        for track_id, track in tracks.items():
            points = track['points']
            if len(points) > 1:
                for i in range(1, len(points)):
                    pt1 = tuple(map(int, points[i - 1][1]))
                    pt2 = tuple(map(int, points[i][1]))
                    cv2.line(mask, pt1, pt2, (0, 255, 0), 2)
                cv2.circle(frame, tuple(map(int, points[-1][1])), 5, (0, 0, 255), -1)
                cv2.putText(frame, str(track_id), tuple(map(int, points[-1][1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        img = cv2.add(frame, mask)

        half_size = int(crosshair_size // 2)
        cv2.line(img, (int(crosshair_x) - half_size, int(crosshair_y)), (int(crosshair_x) + half_size, int(crosshair_y)), (0, 255, 0), 2)
        cv2.line(img, (int(crosshair_x), int(crosshair_y) - half_size), (int(crosshair_x), int(crosshair_y) + half_size), (0, 255, 0), 2)

        debug_out.write(img)
        old_gray = frame_gray.copy()
        pbar.update(1)

    cap.release()
    debug_out.release()
    pbar.close()

    try:
        with open(output_path, 'w') as f:
            json.dump(deltas, f, indent=2)
        print(f"Successfully wrote deltas to {output_path}")
    except Exception as e:
        print(f"Error writing JSON file: {str(e)}")
        print("Returning deltas as a list instead.")

    return deltas