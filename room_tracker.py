import cv2
import numpy as np
import json
from tqdm import tqdm


def detect_persistent_lines(frame, min_length=100, max_line_gap=10):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=min_length, maxLineGap=max_line_gap)
    return lines


def create_line_mask(shape, lines):
    mask = np.zeros(shape, dtype=np.uint8)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(mask, (x1, y1), (x2, y2), 255, 2)
    return mask


def filter_points_near_lines(points, line_mask, distance_threshold=5):
    if points is None or len(points) == 0:
        return np.array([])
    distances = cv2.distanceTransform(255 - line_mask, cv2.DIST_L2, 5)
    near_line = distances[points[:, 1].astype(int), points[:, 0].astype(int)] < distance_threshold
    return points[near_line]


def room_tracker(input_path, output_path, debug_output_path):
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    debug_out = cv2.VideoWriter(debug_output_path, fourcc, fps, (width, height))

    focal_length = 1000
    center = (width // 2, height // 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)

    dist_coeffs = np.zeros((4, 1))

    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=10, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    ret, old_frame = cap.read()
    persistent_lines = detect_persistent_lines(old_frame)
    line_mask = create_line_mask(old_frame.shape[:2], persistent_lines)

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=line_mask, **feature_params)
    if p0 is not None:
        p0 = filter_points_near_lines(p0.reshape(-1, 2), line_mask)
        p0 = p0.reshape(-1, 1, 2)

    mask = np.zeros_like(old_frame)
    deltas = []
    prev_angles = None

    pbar = tqdm(total=total_frames, desc="Processing frames")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if p0 is None or len(p0) < 4:
            persistent_lines = detect_persistent_lines(frame)
            line_mask = create_line_mask(frame.shape[:2], persistent_lines)
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=line_mask, **feature_params)
            if p0 is None:
                frame_count += 1
                pbar.update(1)
                continue
            p0 = filter_points_near_lines(p0.reshape(-1, 2), line_mask)
            p0 = p0.reshape(-1, 1, 2)

        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        if p1 is not None and len(p1) >= 4:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            good_new = filter_points_near_lines(good_new.reshape(-1, 2), line_mask)
            if len(good_new) >= 4:
                obj_points = np.hstack((good_new, np.zeros((good_new.shape[0], 1)))).astype(np.float32)

                success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
                    obj_points, good_new, camera_matrix, dist_coeffs
                )

                if success:
                    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                    euler_angles = cv2.decomposeProjectionMatrix(np.hstack((rotation_matrix, translation_vector)))[6]

                    pan, tilt, roll = [angle[0] for angle in euler_angles]

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

                # Draw the tracks and lines
                frame_with_lines = frame.copy()
                if persistent_lines is not None:
                    for line in persistent_lines:
                        x1, y1, x2, y2 = line[0]
                        cv2.line(frame_with_lines, (x1, y1), (x2, y2), (0, 255, 255), 2)

                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                    frame_with_lines = cv2.circle(frame_with_lines, (int(a), int(b)), 5, (0, 0, 255), -1)

                img = cv2.add(frame_with_lines, mask)

                if prev_angles:
                    cv2.putText(img, f"Pan: {pan:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, f"Tilt: {tilt:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, f"Roll: {roll:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                debug_out.write(img)

                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)
            else:
                p0 = cv2.goodFeaturesToTrack(frame_gray, mask=line_mask, **feature_params)
                if p0 is not None:
                    p0 = filter_points_near_lines(p0.reshape(-1, 2), line_mask)
                    p0 = p0.reshape(-1, 1, 2)
        else:
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=line_mask, **feature_params)
            if p0 is not None:
                p0 = filter_points_near_lines(p0.reshape(-1, 2), line_mask)
                p0 = p0.reshape(-1, 1, 2)

        frame_count += 1
        pbar.update(1)

    cap.release()
    debug_out.release()
    pbar.close()

    with open(output_path, 'w') as f:
        json.dump(deltas, f, indent=2)

    return deltas

