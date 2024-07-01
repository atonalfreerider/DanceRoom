import cv2
import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import KMeans

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    
    room_geometry, camera_params = reconstruct_room_and_camera(frames)
    return room_geometry, camera_params

def reconstruct_room_and_camera(frames):
    initial_geometry = estimate_initial_geometry(frames[0])
    initial_camera = estimate_initial_camera(frames[0])
    
    def objective_function(params):
        room_params = params[:6]  # [width, length, height, x, y, z]
        camera_params = params[6:]  # [fx, fy, cx, cy, rx, ry, rz, tx, ty, tz]
        
        error = 0
        for frame in frames:
            projected_geometry = project_geometry(room_params, camera_params)
            error += compute_error(frame, projected_geometry)
        
        return error
    
    initial_params = np.concatenate([initial_geometry, initial_camera])
    result = minimize(objective_function, initial_params, method='Powell')
    
    optimized_params = result.x
    room_geometry = optimized_params[:6]
    camera_params = optimized_params[6:]
    
    return room_geometry, camera_params

def estimate_initial_geometry(frame):
    edges = cv2.Canny(frame, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    vanishing_points = estimate_vanishing_points(lines)
    initial_geometry = estimate_room_dimensions(vanishing_points)
    
    return initial_geometry

def estimate_initial_camera(frame):
    height, width = frame.shape
    focal_length = width  # Approximation
    cx, cy = width / 2, height / 2
    
    # Initial camera parameters: [fx, fy, cx, cy, rx, ry, rz, tx, ty, tz]
    return np.array([focal_length, focal_length, cx, cy, 0, 0, 0, 0, 0, 0])

def project_geometry(room_params, camera_params):
    width, length, height, x, y, z = room_params
    fx, fy, cx, cy, rx, ry, rz, tx, ty, tz = camera_params
    
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
    
    # Create rotation matrix
    R = cv2.Rodrigues(np.array([rx, ry, rz]))[0]
    
    # Create camera matrix
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    
    # Project 3D points to 2D
    points_2d, _ = cv2.projectPoints(points_3d, R, np.array([tx, ty, tz]), K, None)
    
    return points_2d.reshape(-1, 2)

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
        
        px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / denom
        
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
    return np.array([width, length, height, -width/2, -length/2, -height/2])

# Main execution
video_path = "/home/john/Downloads/carlos-aline-spin-cam1.mp4"
room_geometry, camera_params = process_video(video_path)
print("Room Geometry:", room_geometry)
print("Camera Parameters:", camera_params)
