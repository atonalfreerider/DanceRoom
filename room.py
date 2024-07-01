import cv2
import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from filterpy.kalman import KalmanFilter


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()

    room_geometry, camera_params, pan_angles, zoom_values = reconstruct_scene(frames)
    return room_geometry, camera_params, pan_angles, zoom_values


def reconstruct_scene(frames):
    initial_geometry = estimate_initial_geometry(frames[0])
    initial_camera = estimate_initial_camera(frames[0])
    initial_pan = np.zeros(len(frames))
    initial_zoom = np.ones(len(frames))

    def objective_function(params):
        room_params = params[:6]  # [width, length, height, x, y, z]
        camera_params = params[6:13]  # [fx, fy, cx, cy, cam_x, cam_y, cam_z]
        pan_angles = params[13:13 + len(frames)]
        zoom_values = params[13 + len(frames):]

        error = 0
        for i, frame in enumerate(frames):
            projected_geometry = project_geometry(room_params, camera_params, pan_angles[i], zoom_values[i])
            error += compute_error(frame, projected_geometry)

        # Add regularization terms
        error += 0.1 * np.sum(np.diff(pan_angles) ** 2)  # Smoothness of pan angles
        error += 10 * np.sum(np.diff(zoom_values) ** 2)  # Encourage step-like zoom function

        return error

    initial_params = np.concatenate([initial_geometry, initial_camera, initial_pan, initial_zoom])

    # Add constraints
    constraints = [
        {'type': 'ineq', 'fun': lambda x: x[12] - 0},  # cam_z >= 0
        {'type': 'ineq', 'fun': lambda x: 2 - x[12]},  # cam_z <= 2
    ]

    result = minimize(objective_function, initial_params, method='SLSQP', constraints=constraints)

    optimized_params = result.x
    room_geometry = optimized_params[:6]
    camera_params = optimized_params[6:13]
    pan_angles = optimized_params[13:13 + len(frames)]
    zoom_values = optimized_params[13 + len(frames):]

    # Apply temporal smoothing
    pan_angles = smooth_pan_angles(pan_angles)
    zoom_values = smooth_zoom_values(zoom_values)

    return room_geometry, camera_params, pan_angles, zoom_values


def estimate_initial_geometry(frame):
    edges = cv2.Canny(frame, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    vanishing_points = estimate_vanishing_points(lines)
    initial_geometry = estimate_room_dimensions(vanishing_points)

    return initial_geometry


def estimate_initial_camera(frame):
    height, width = frame.shape
    focal_length = width  # Approximation
    cx, cy = width / 2, height / 2

    # Initial camera parameters: [fx, fy, cx, cy, cam_x, cam_y, cam_z]
    return np.array([focal_length, focal_length, cx, cy, 0, 0, 1])


def project_geometry(room_params, camera_params, pan_angle, zoom):
    width, length, height, x, y, z = room_params
    fx, fy, cx, cy, cam_x, cam_y, cam_z = camera_params

    # Create 3D points for room corners
    points_3d = np.array([
        [x, y, z],
        [x + width, y, z],
        [x + width, y + length, z],
        [x, y + length, z],
        [x, y, z + height],
        [x + width, y, z + height],
        [x + width, y + length, z + height],
        [x, y + length, z + height]
    ])

    # Apply camera position
    points_3d -= np.array([cam_x, cam_y, cam_z])

    # Create rotation matrix for pan
    R = cv2.Rodrigues(np.array([0, pan_angle, 0]))[0]

    # Apply rotation
    points_3d = np.dot(points_3d, R.T)

    # Create camera matrix with zoom
    K = np.array([[fx * zoom, 0, cx], [0, fy * zoom, cy], [0, 0, 1]])

    # Project 3D points to 2D
    points_2d = np.dot(points_3d, K.T)
    points_2d = points_2d[:, :2] / points_2d[:, 2:]

    return points_2d


def compute_error(frame, projected_geometry):
    height, width = frame.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    # Draw projected geometry on mask
    for i in range(4):
        pt1 = tuple(map(int, projected_geometry[i]))
        pt2 = tuple(map(int, projected_geometry[(i + 1) % 4]))
        cv2.line(mask, pt1, pt2, 255, 2)
        pt3 = tuple(map(int, projected_geometry[i + 4]))
        cv2.line(mask, pt1, pt3, 255, 2)

    for i in range(4, 8):
        pt1 = tuple(map(int, projected_geometry[i]))
        pt2 = tuple(map(int, projected_geometry[(i - 3) % 4 + 4]))
        cv2.line(mask, pt1, pt2, 255, 2)

    # Compute error
    edges = cv2.Canny(frame, 50, 150)
    error = np.sum(np.abs(edges.astype(float) - mask.astype(float)))

    return error


def estimate_vanishing_points(lines):
    def line_intersection(line1, line2):
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            return None

        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

        return px, py

    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            intersection = line_intersection(lines[i][0], lines[j][0])
            if intersection:
                intersections.append(intersection)

    # Cluster intersections to find vanishing points
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(intersections)
    vanishing_points = kmeans.cluster_centers_

    return vanishing_points


def estimate_room_dimensions(vanishing_points):
    # This is a simplified estimation and may need refinement
    vp1, vp2, vp3 = vanishing_points

    # Estimate room width, length, and height based on vanishing point distances
    width = np.linalg.norm(vp1 - vp2)
    length = np.linalg.norm(vp2 - vp3)
    height = np.linalg.norm(vp3 - vp1)

    # Normalize dimensions
    total = width + length + height
    width /= total
    length /= total
    height /= total

    # Assume room center is at (0, 0, 0)
    return np.array([width, length, height, -width / 2, -length / 2, -height / 2])


def smooth_pan_angles(pan_angles):
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([0., 0.])  # Initial state (angle and angular velocity)
    kf.F = np.array([[1., 1.], [0., 1.]])  # State transition matrix
    kf.H = np.array([[1., 0.]])  # Measurement function
    kf.P *= 1000.  # Covariance matrix
    kf.R = 0.1  # Measurement noise
    kf.Q = np.array([[0.01, 0.01], [0.01, 0.1]])  # Process noise

    smoothed_angles = []
    for angle in pan_angles:
        kf.predict()
        kf.update(angle)
        smoothed_angles.append(kf.x[0])

    return np.array(smoothed_angles)


def smooth_zoom_values(zoom_values):
    # Use total variation denoising for step-like function
    from skimage.restoration import denoise_tv_chambolle

    smoothed_zoom = denoise_tv_chambolle(zoom_values, weight=0.1)
    return smoothed_zoom


# Main execution
video_path = "/home/john/Downloads/carlos-aline-spin-cam1.mp4"
room_geometry, camera_params, pan_angles, zoom_values = process_video(video_path)
print("Room Geometry:", room_geometry)
print("Camera Parameters:", camera_params)
print("Pan Angles:", pan_angles)
print("Zoom Values:", zoom_values)
