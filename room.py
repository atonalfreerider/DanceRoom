import cv2
import numpy as np
import time
import os
import json
from scipy.optimize import minimize
from sklearn.cluster import DBSCAN

class CameraModel:
    def __init__(self, focal_length, principal_point, distortion_coeffs):
        self.focal_length = focal_length
        self.principal_point = principal_point
        self.distortion_coeffs = distortion_coeffs

    def project(self, points_3d):
        if len(points_3d.shape) == 1:
            points_3d = points_3d.reshape(1, -1)

        x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]

        # Add a small epsilon to avoid division by zero
        z = np.where(z == 0, 1e-10, z)

        x_prime = x / z
        y_prime = y / z

        r2 = x_prime ** 2 + y_prime ** 2

        fx, fy = self.focal_length
        cx, cy = self.principal_point
        k1, k2, p1, p2, k3 = self.distortion_coeffs

        radial_distortion = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        tangential_x = 2 * p1 * x_prime * y_prime + p2 * (r2 + 2 * x_prime ** 2)
        tangential_y = p1 * (r2 + 2 * y_prime ** 2) + 2 * p2 * x_prime * y_prime

        x_distorted = x_prime * radial_distortion + tangential_x
        y_distorted = y_prime * radial_distortion + tangential_y

        u = fx * x_distorted + cx
        v = fy * y_distorted + cy

        return np.column_stack((u, v))

    def unproject(self, points_2d, depth):
        fx, fy = self.focal_length
        cx, cy = self.principal_point

        x = (points_2d[:, 0] - cx) / fx
        y = (points_2d[:, 1] - cy) / fy

        points_3d = np.column_stack((x * depth, y * depth, depth))
        return points_3d

    def optimize_params(self, points_2d, points_3d):
        def objective(params):
            self.focal_length = params[:2]
            self.principal_point = params[2:4]
            self.distortion_coeffs = params[4:]
            projected = self.project(points_3d)
            return np.sum((points_2d - projected) ** 2)

        initial_params = np.hstack((self.focal_length, self.principal_point, self.distortion_coeffs))

        print("Starting camera parameter optimization")
        start_time = time.time()

        def callback(xk):
            nonlocal start_time
            elapsed_time = time.time() - start_time
            error = objective(xk)
            print(f"Camera optimization: Error = {error:.2f}, Time: {elapsed_time:.2f}s")

        result = minimize(objective, initial_params, method='Powell', callback=callback)

        self.focal_length = result.x[:2]
        self.principal_point = result.x[2:4]
        self.distortion_coeffs = result.x[4:]

        print("Camera parameter optimization completed")

    def get_params(self):
        return np.hstack((self.focal_length, self.principal_point, self.distortion_coeffs))


class RoomModel:
    def __init__(self, geometry, pan_angles, zoom_values):
        self.geometry = geometry  # [width, length, height, x, y, z]
        self.pan_angles = pan_angles
        self.zoom_values = zoom_values


def extract_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    edges = cv2.Canny(binary, 30, 100)  # Adjusted thresholds
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=30, maxLineGap=20)  # Adjusted parameters
    corners = cv2.goodFeaturesToTrack(gray, 200, 0.005, 10)  # Increased number of corners, lowered quality level
    if corners is not None:
        corners = corners.reshape(-1, 2).astype(np.int32)
    else:
        corners = np.array([]).reshape(0, 2)
    return edges, lines, corners


def cluster_orientations(orientations):
    orientations = orientations.reshape(-1, 1)
    clustering = DBSCAN(eps=0.1, min_samples=5).fit(orientations)
    return clustering.labels_


def find_dominant_orientations(clusters):
    unique_labels, counts = np.unique(clusters, return_counts=True)
    dominant_indices = np.argsort(counts)[-3:]  # Get indices of top 3 clusters
    return unique_labels[dominant_indices]


def group_lines_by_orientation(lines, dominant_orientations):
    orientations = np.array([np.arctan2(x2 - x1, y2 - y1) for (x1, y1, x2, y2) in lines[:, 0]])
    grouped_lines = [[] for _ in range(len(dominant_orientations))]
    for line, orientation in zip(lines, orientations):
        closest_dominant = dominant_orientations[np.argmin(np.abs(orientation - dominant_orientations))]
        grouped_lines[np.where(dominant_orientations == closest_dominant)[0][0]].append(line)
    return grouped_lines


def estimate_vanishing_points(grouped_lines):
    vanishing_points = []
    for group in grouped_lines:
        if len(group) < 2:
            continue
        A = np.zeros((len(group), 2))
        b = np.zeros((len(group), 1))
        for i, line in enumerate(group):
            x1, y1, x2, y2 = line[0]
            A[i] = [y2 - y1, x1 - x2]
            b[i] = x1 * y2 - x2 * y1
        vp = np.linalg.lstsq(A, b, rcond=None)[0]
        vanishing_points.append(vp.flatten())
    return np.array(vanishing_points)


def estimate_dimensions_from_vp(vanishing_points, corners):
    # This is a simplified estimation and may need refinement
    if len(vanishing_points) < 2:
        print("Warning: Not enough vanishing points detected. Using fallback method.")
        # Fallback method: use default values if not enough corners
        if len(corners) < 2:
            print("Warning: Not enough corners detected. Using default values.")
            width, length, height = 1.0, 1.0, 1.0
            center = [0, 0]
        else:
            min_corner = np.min(corners, axis=0)
            max_corner = np.max(corners, axis=0)
            width = max_corner[0] - min_corner[0]
            height = max_corner[1] - min_corner[1]
            length = (width + height) / 2  # Rough estimate
            center = (min_corner + max_corner) / 2
    elif len(vanishing_points) == 2:
        vp1, vp2 = vanishing_points
        width = np.linalg.norm(vp1 - vp2)
        height = width * 0.75  # Assuming a typical room height
        length = (width + height) / 2
        center = np.mean(corners, axis=0) if len(corners) > 0 else [0, 0]
    else:
        vp1, vp2, vp3 = vanishing_points
        width = np.linalg.norm(vp1 - vp2)
        length = np.linalg.norm(vp2 - vp3)
        height = np.linalg.norm(vp3 - vp1)
        center = np.mean(corners, axis=0) if len(corners) > 0 else [0, 0]

    # Normalize dimensions
    total = width + length + height
    width /= total
    length /= total
    height /= total

    return np.array([width, length, height, center[0], center[1], 0])


def estimate_room_geometry(lines, corners):
    if lines is None or len(lines) < 2:
        print("Warning: Not enough lines detected. Using fallback method.")
        return estimate_dimensions_from_vp([], corners)

    orientations = np.array([np.arctan2(x2 - x1, y2 - y1) for (x1, y1, x2, y2) in lines[:, 0]])
    orientation_clusters = cluster_orientations(orientations)
    dominant_orientations = find_dominant_orientations(orientation_clusters)
    grouped_lines = group_lines_by_orientation(lines, dominant_orientations)
    vanishing_points = estimate_vanishing_points(grouped_lines)
    room_dimensions = estimate_dimensions_from_vp(vanishing_points, corners)
    return room_dimensions


def random_sample(features, n):
    indices = np.random.choice(len(features), n, replace=False)
    return [features[i] for i in indices]


def estimate_model(sample, camera_model):
    # Unpack the sample
    edges, lines, corners = sample
    if lines is None or len(lines) < 2 or len(corners) < 4:
        # Not enough information to estimate a model
        return None

    # Use the existing estimate_room_geometry function
    geometry = estimate_room_geometry(lines, corners)

    # Create and return a RoomModel
    return RoomModel(geometry, np.zeros(1), np.ones(1))


def model_error(feature, model, camera_model):
    edges, lines, corners = feature
    if lines is None or len(lines) < 2:
        return float('inf')  # Return a large error if we don't have enough information

    # Project the model's corners using the camera model
    projected_corners = camera_model.project(model.geometry[:3])

    # Compare the projected corners with the actual corners
    error = np.mean(np.min(np.linalg.norm(projected_corners[:, None] - corners[None, :], axis=2), axis=1))

    return error


def refine_model(model, inliers, camera_model):
    # Extract all corners and lines from inliers
    all_corners = []
    all_lines = []
    for edges, lines, corners in inliers:
        if corners is not None and len(corners) > 0:
            all_corners.extend(corners)
        if lines is not None and len(lines) > 0:
            all_lines.extend(lines)

    all_corners = np.array(all_corners)
    all_lines = np.array(all_lines)

    # Estimate room geometry using all inlier features
    refined_geometry = estimate_room_geometry(all_lines, all_corners)

    # Project 3D room corners to 2D using the camera model
    room_corners_3d = np.array([
        [0, 0, 0],
        [refined_geometry[0], 0, 0],
        [refined_geometry[0], refined_geometry[1], 0],
        [0, refined_geometry[1], 0],
        [0, 0, refined_geometry[2]],
        [refined_geometry[0], 0, refined_geometry[2]],
        [refined_geometry[0], refined_geometry[1], refined_geometry[2]],
        [0, refined_geometry[1], refined_geometry[2]]
    ])
    projected_corners = camera_model.project(room_corners_3d)

    # Optimize room dimensions to minimize reprojection error
    def objective(params):
        width, length, height = params
        room_corners = np.array([
            [0, 0, 0],
            [width, 0, 0],
            [width, length, 0],
            [0, length, 0],
            [0, 0, height],
            [width, 0, height],
            [width, length, height],
            [0, length, height]
        ])
        projected = camera_model.project(room_corners)
        error = np.sum((projected - projected_corners) ** 2)
        return error

    result = minimize(objective, refined_geometry[:3], method='Powell')
    optimized_geometry = np.concatenate([result.x, refined_geometry[3:]])

    # Update pan angles and zoom values based on inliers
    # (This is a placeholder - you might want to implement a more sophisticated method)
    pan_angles = model.pan_angles  # For now, keep the original pan angles
    zoom_values = model.zoom_values  # For now, keep the original zoom values

    return RoomModel(optimized_geometry, pan_angles, zoom_values)


def ransac_room_estimation(features, camera_model, num_iterations=5000, threshold=0.1, max_attempts=3):
    for attempt in range(max_attempts):
        best_model = None
        best_inliers = []

        print(f"Starting RANSAC attempt {attempt + 1} with {num_iterations} iterations")
        start_time = time.time()

        for i in range(num_iterations):
            sample = random_sample(features, 1)[0]  # Take one complete feature set
            model = estimate_model(sample, camera_model)

            if model is None:
                continue  # Skip this iteration if we couldn't estimate a model

            inliers = [f for f in features if model_error(f, model, camera_model) < threshold]

            if len(inliers) > len(best_inliers):
                best_model = model
                best_inliers = inliers

            if i % 100 == 0:  # Print every 100 iterations
                elapsed_time = time.time() - start_time
                print(f"RANSAC Iteration {i}: Best inliers = {len(best_inliers)}, Time: {elapsed_time:.2f}s")

        if best_model is not None:
            print(f"RANSAC completed. Best model has {len(best_inliers)} inliers")
            refined_model = refine_model(best_model, best_inliers, camera_model)
            return refined_model, best_inliers

        print(f"RANSAC attempt {attempt + 1} failed. Adjusting parameters and trying again.")
        threshold *= 1.5  # Increase threshold for next attempt
        num_iterations = int(num_iterations * 1.5)  # Increase number of iterations for next attempt

    print("All RANSAC attempts failed to find a valid model")
    return None, []


def estimate_initial_model(features):
    edges, lines, corners = features[0]
    geometry = estimate_room_geometry(lines, corners)
    return RoomModel(geometry, np.zeros(1), np.ones(1))


def refine_model(initial_model, features):
    refined_geometry = np.zeros_like(initial_model.geometry)
    for edges, lines, corners in features:
        geometry = estimate_room_geometry(lines, corners)
        refined_geometry += geometry
    refined_geometry /= len(features)
    return RoomModel(refined_geometry, initial_model.pan_angles, initial_model.zoom_values)


def hierarchical_optimization(frames):
    print("Starting hierarchical optimization")

    print("Coarse optimization")
    coarse_features = [extract_features(frame) for frame in frames[::10]]
    coarse_model = estimate_initial_model(coarse_features)
    print(f"Coarse model geometry: {coarse_model.geometry}")

    print("Medium optimization")
    medium_features = [extract_features(frame) for frame in frames[::5]]
    medium_model = refine_model(coarse_model, medium_features)
    print(f"Medium model geometry: {medium_model.geometry}")

    print("Fine optimization")
    fine_features = [extract_features(frame) for frame in frames]
    fine_model = refine_model(medium_model, fine_features)
    print(f"Fine model geometry: {fine_model.geometry}")

    return fine_model


def reconstruct_scene(frames):
    print("Starting scene reconstruction")
    all_features = [extract_features(frame) for frame in frames]
    print(f"Extracted features from {len(frames)} frames")

    height, width = frames[0].shape[:2]
    camera_model = CameraModel(
        focal_length=(width, width),
        principal_point=(width / 2, height / 2),
        distortion_coeffs=np.zeros(5)
    )

    print("Starting hierarchical optimization")
    room_model = hierarchical_optimization(frames)

    # Dummy 2D and 3D points for camera optimization
    points_2d = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=float)
    points_3d = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    camera_model.optimize_params(points_2d, points_3d)

    print("Starting RANSAC room estimation")
    final_model, inliers = ransac_room_estimation(all_features, camera_model, num_iterations=5000)

    if final_model is None:
        print("RANSAC failed. Using the result from hierarchical optimization.")
        final_model = room_model

    room_geometry = final_model.geometry
    camera_params = camera_model.get_params()
    pan_angles = final_model.pan_angles
    zoom_values = final_model.zoom_values

    print("Scene reconstruction completed")
    return room_geometry, camera_params, pan_angles, zoom_values


def draw_features(frame, edges, lines, corners, vanishing_points):
    result = frame.copy()

    # Draw edges
    result[edges != 0] = [0, 255, 0]

    # Draw lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Draw corners
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(result, (x, y), 3, (255, 0, 0), -1)

    # Draw vanishing points
    for vp in vanishing_points:
        x, y = int(vp[0]), int(vp[1])
        cv2.circle(result, (x, y), 10, (255, 255, 0), -1)

    return result


def process_video(video_path, output_path):
    print(f"Processing video: {video_path}")

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % int(fps) == 0:  # Sample one frame per second
            frames.append(frame)
        frame_count += 1

    cap.release()
    print(f"Loaded {len(frames)} frames from video")

    all_features = []
    for i, frame in enumerate(frames):
        edges, lines, corners = extract_features(frame)
        all_features.append((edges, lines, corners))

        # Estimate room geometry for visualization
        if lines is not None and len(lines) >= 2:
            orientations = np.array([np.arctan2(x2 - x1, y2 - y1) for (x1, y1, x2, y2) in lines[:, 0]])
            orientation_clusters = cluster_orientations(orientations)
            dominant_orientations = find_dominant_orientations(orientation_clusters)
            grouped_lines = group_lines_by_orientation(lines, dominant_orientations)
            vanishing_points = estimate_vanishing_points(grouped_lines)
        else:
            vanishing_points = []

        # Draw features on frame
        result_frame = draw_features(frame, edges, lines, corners, vanishing_points)

        # Save the frame with drawn features
        cv2.imwrite(os.path.join(output_path, f"frame_{i:04d}.jpg"), result_frame)

    room_geometry, camera_params, pan_angles, zoom_values = reconstruct_scene(frames)

    # Save results to JSON files
    results = {
        "room_geometry": room_geometry.tolist(),
        "camera_params": camera_params.tolist(),
        "pan_angles": pan_angles.tolist(),
        "zoom_values": zoom_values.tolist()
    }

    with open(os.path.join(output_path, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    return room_geometry, camera_params, pan_angles, zoom_values


# Main execution
if __name__ == "__main__":
    video_path = "/home/john/Downloads/carlos-aline-spin-cam2.mp4"
    output_path = "/home/john/Desktop/room_reconstruction_output"

    print("Starting room reconstruction from video")
    room_geometry, camera_params, pan_angles, zoom_values = process_video(video_path, output_path)

    print("\nFinal Results:")
    print("Room Geometry:", room_geometry)
    print("Camera Parameters:", camera_params)
    print("Pan Angles:", pan_angles)
    print("Zoom Values:", zoom_values)
    print("Room reconstruction completed")
    print(f"Results and visualizations saved to {output_path}")
