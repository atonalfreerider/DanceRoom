import cv2
import numpy as np
import json
from tqdm import tqdm
from scipy.spatial import distance


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


def estimate_transform(src_points, dst_points):
    # Use RANSAC to estimate the transformation
    transform_matrix, inliers = cv2.estimateAffinePartial2D(src_points, dst_points, method=cv2.RANSAC,
                                                            ransacReprojThreshold=3.0)

    if transform_matrix is not None:
        # Extract translation
        dx, dy = transform_matrix[:2, 2]

        # Extract rotation
        rotation = np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])

        # Extract scale (zoom)
        scale = np.sqrt(transform_matrix[0, 0] ** 2 + transform_matrix[1, 0] ** 2)

        return dx, dy, rotation, scale, inliers
    else:
        return 0, 0, 0, 1, None


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

    # Initialize feature detector and tracker
    feature_params = dict(qualityLevel=0.01, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Read first frame and detect initial features
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # Create a mask to exclude logo area (adjust coordinates as needed)
    logo_mask = np.ones(old_gray.shape[:2], dtype=np.uint8)
    logo_mask[0:50, 0:50] = 0  # Assume logo is in top-left corner, adjust as needed

    # Initialize variables for storing deltas and tracks
    deltas = []
    tracks = {}
    next_track_id = 0
    max_track_length = int(5 * fps)  # 5 seconds of tracks

    # Initialize crosshair position
    crosshair_x = width // 2
    crosshair_y = height // 2
    initial_crosshair_size = 20
    crosshair_size = initial_crosshair_size

    frame_count = 0
    pbar = tqdm(total=total_frames, desc="Processing frames")

    prev_points = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Get YOLO detections for the current frame
        current_detections = yolo_detections.get(str(frame_count), [])
        current_boxes = [detection["bbox"] for detection in current_detections]

        good_new = []
        good_old = []

        # Calculate optical flow for existing tracks
        if tracks:
            prev_points = np.float32([track['points'][-1][1] for track in tracks.values()]).reshape(-1, 1, 2)
            curr_points, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, prev_points, None, **lk_params)

            # Update tracks
            updated_tracks = {}

            for (track_id, track), (new, status) in zip(list(tracks.items()), zip(curr_points, st.ravel())):
                if status:
                    if not any(point_in_box(new[0], box) for box in current_boxes):
                        track['points'].append((frame_count, tuple(new.ravel())))
                        track['last_seen'] = frame_count
                        while track['points'] and frame_count - track['points'][0][0] >= max_track_length:
                            track['points'].pop(0)
                        updated_tracks[track_id] = track
                        good_new.append(new.reshape(2))
                        good_old.append(prev_points[list(tracks.keys()).index(track_id)].reshape(2))

            tracks = updated_tracks

        # Detect new points if needed
        if len(tracks) < 20:
            mask = np.ones(frame_gray.shape[:2], dtype=np.uint8)

            # Mask out existing tracks
            for track in tracks.values():
                x, y = map(int, track['points'][-1][1])
                cv2.circle(mask, (x, y), 10, 0, -1)

            # Mask out YOLO detection boxes
            for box in current_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)

            new_points = cv2.goodFeaturesToTrack(frame_gray, mask=mask,
                                                 maxCorners=20 - len(tracks), **feature_params)
            if new_points is not None:
                for point in new_points:
                    tracks[next_track_id] = {
                        'points': [(frame_count, tuple(point.ravel()))],
                        'last_seen': frame_count
                    }
                    next_track_id += 1

        good_new = np.array(good_new)
        good_old = np.array(good_old)

        # Estimate transform and calculate deltas
        if len(good_new) >= 4 and len(good_old) >= 4:
            dx, dy, rotation, scale, inliers = estimate_transform(good_old, good_new)

            if inliers is not None:
                # Use only inlier points
                good_new = good_new[inliers.ravel() == 1]
                good_old = good_old[inliers.ravel() == 1]

                # Calculate zoom
                zoom_factor = scale
                delta_zoom = zoom_factor - 1

                deltas.append({
                    "frame": frame_count,
                    "delta_x": float(dx),
                    "delta_y": float(dy),
                    "delta_roll": float(rotation),
                    "delta_zoom": float(delta_zoom)
                })

                # Update crosshair position and size
                crosshair_x -= int(dx)
                crosshair_y -= int(dy)
                crosshair_size *= zoom_factor

                # Rotate crosshair
                rot_matrix = cv2.getRotationMatrix2D((crosshair_x, crosshair_y), np.degrees(rotation), 1)
                crosshair_points = np.array([
                    [crosshair_x - crosshair_size, crosshair_y],
                    [crosshair_x + crosshair_size, crosshair_y],
                    [crosshair_x, crosshair_y - crosshair_size],
                    [crosshair_x, crosshair_y + crosshair_size]
                ])
                rotated_crosshair = cv2.transform(np.array([crosshair_points]), rot_matrix)[0]

                # Reset crosshair if it moves out of frame
                if (crosshair_x < 0 or crosshair_x >= width or
                        crosshair_y < 0 or crosshair_y >= height):
                    crosshair_x = width // 2
                    crosshair_y = height // 2
                    crosshair_size = initial_crosshair_size

        # Draw tracks and crosshair
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

        # Draw crosshair
        half_size = int(crosshair_size // 2)
        cv2.line(img, (crosshair_x - half_size, crosshair_y), (crosshair_x + half_size, crosshair_y), (0, 255, 0), 2)
        cv2.line(img, (crosshair_x, crosshair_y - half_size), (crosshair_x, crosshair_y + half_size), (0, 255, 0), 2)

        debug_out.write(img)

        old_gray = frame_gray.copy()

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

