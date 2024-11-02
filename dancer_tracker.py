import cv2
import numpy as np
import json
import os
import tqdm
from deepface import DeepFace
import shutil
from collections import defaultdict
import random

# uses DeepFace to find self-similar faces and gender
class DancerTracker:
    def __init__(self, input_path, output_dir):
        self.input_path = input_path
        self.output_dir = output_dir
        self.depth_dir = os.path.join(output_dir, 'depth')
        self.figure_mask_dir = os.path.join(output_dir, 'figure-masks')
        
        # New directories for face processing
        self.faces_dir = os.path.join(output_dir, 'faces')
        self.person_dirs = os.path.join(output_dir, 'person_clusters')
        self.lead_faces_dir = os.path.join(output_dir, 'lead_faces')
        self.follow_faces_dir = os.path.join(output_dir, 'follow_faces')
        
        # Create necessary directories
        for dir_path in [self.faces_dir, self.person_dirs, 
                        self.lead_faces_dir, self.follow_faces_dir]:
            os.makedirs(dir_path, exist_ok=True)

        self.detections_file = os.path.join(output_dir, 'detections.json')
        self.lead_file = os.path.join(output_dir, 'lead.json')
        self.follow_file = os.path.join(output_dir, 'follow.json')

        self.detections = self.load_json(self.detections_file)
        
        # Get input video dimensions
        cap = cv2.VideoCapture(input_path)
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()

    def process_video(self):
        # Check if faces directory already has crops
        existing_faces = os.listdir(self.faces_dir)
        if not existing_faces:
            self.extract_face_crops()
        else:
            print(f"Found {len(existing_faces)} existing face crops, skipping extraction...")
        
        self.analyze_and_sort_by_gender()
        self.create_role_assignments()
        self.interpolate_missing_assignments()
        print("Lead and follow tracked using DeepFace approach")

    def extract_face_crops(self):
        """Step 1: Extract face crops from poses meeting height threshold"""
        print("Extracting face crops...")
        frame_count = len(os.listdir(self.figure_mask_dir))
        min_height_threshold = 0.6 * self.frame_height
        
        for frame_num in tqdm.tqdm(range(frame_count)):
            frame_path = os.path.join(self.figure_mask_dir, f"{frame_num:06d}.png")
            if not os.path.exists(frame_path):
                continue
                
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
                
            detections_in_frame = self.detections.get(frame_num, [])
            
            for detection in detections_in_frame:
                bbox = detection['bbox']
                height = bbox[3] - bbox[1]
                
                if height >= min_height_threshold:
                    head_bbox = self.get_head_bbox(detection['keypoints'])
                    if head_bbox:
                        x1, y1, x2, y2 = head_bbox
                        head_img = frame[y1:y2, x1:x2]
                        
                        if head_img.size > 0:
                            output_path = os.path.join(
                                self.faces_dir, 
                                f"{frame_num:06d}-{detection['id']}.jpg"
                            )
                            cv2.imwrite(output_path, head_img)

    def get_head_bbox(self, keypoints, padding_percent=0.25):
        """Extract square head bounding box from keypoints with padding"""
        # Get head keypoints (nose, eyes, ears) - indices 0-4
        head_points = keypoints[:5]
        
        # Filter out low confidence or missing points (0,0 coordinates)
        valid_points = [point for point in head_points
                        if point[2] > 0.3 and (point[0] != 0 or point[1] != 0)]
        
        if not valid_points:
            return None
            
        # Convert to numpy array for easier computation
        points = np.array(valid_points)[:, :2]  # Only take x,y coordinates
        
        # Get bounding box
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        
        # Calculate center point
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        # Get the larger dimension for square crop
        width = x_max - x_min
        height = y_max - y_min
        size = max(width, height)
        
        # Add padding
        size_with_padding = size * (1 + 2 * padding_percent)
        half_size = size_with_padding / 2
        
        # Calculate square bounds from center
        x_min = center_x - half_size
        x_max = center_x + half_size
        y_min = center_y - half_size
        y_max = center_y + half_size
        
        # Ensure bounds are within frame
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(self.frame_width, x_max)
        y_max = min(self.frame_height, y_max)
        
        return [int(x_min), int(y_min), int(x_max), int(y_max)]

    def analyze_and_sort_by_gender(self):
        """Analyze each face for gender and sort into lead/follow folders"""
        print("Analyzing faces for gender...")
        
        # Clear existing role directories
        for dir_path in [self.lead_faces_dir, self.follow_faces_dir]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
            os.makedirs(dir_path)
        
        face_files = os.listdir(self.faces_dir)
        
        if not face_files:
            raise Exception("No face crops found!")
        
        for face_file in tqdm.tqdm(face_files):
            try:
                img_path = os.path.join(self.faces_dir, face_file)
                result = DeepFace.analyze(
                    img_path=img_path,
                    actions=['gender'],
                    enforce_detection=False
                )
                
                if isinstance(result, list):
                    result = result[0]
                
                # Copy to appropriate folder based on gender
                if result['dominant_gender'] == 'Man':
                    dst = os.path.join(self.lead_faces_dir, face_file)
                else:  # 'Woman'
                    dst = os.path.join(self.follow_faces_dir, face_file)
                    
                shutil.copy2(img_path, dst)
                
            except Exception as e:
                print(f"Error analyzing {face_file}: {str(e)}")
                continue
        
        # Print statistics
        lead_count = len(os.listdir(self.lead_faces_dir))
        follow_count = len(os.listdir(self.follow_faces_dir))
        print(f"Sorted {lead_count} male (lead) and {follow_count} female (follow) faces")

    def create_role_assignments(self):
        """Create lead and follow assignments based on gender analysis"""
        lead_poses = {}
        follow_poses = {}
        
        # Parse filenames to get frame numbers and track IDs
        lead_assignments = self._parse_role_files(self.lead_faces_dir)
        follow_assignments = self._parse_role_files(self.follow_faces_dir)
        
        # Create pose assignments
        for frame_num in self.detections:
            detections_in_frame = self.detections[frame_num]
            
            # Assign lead
            if frame_num in lead_assignments:
                lead_id = lead_assignments[frame_num]
                lead_pose = next(
                    (d for d in detections_in_frame if d['id'] == lead_id),
                    None
                )
                if lead_pose:
                    lead_poses[str(frame_num)] = {
                        'id': lead_pose['id'],
                        'bbox': lead_pose['bbox'],
                        'confidence': lead_pose['confidence'],
                        'keypoints': lead_pose['keypoints']
                    }
            
            # Assign follow
            if frame_num in follow_assignments:
                follow_id = follow_assignments[frame_num]
                follow_pose = next(
                    (d for d in detections_in_frame if d['id'] == follow_id),
                    None
                )
                if follow_pose:
                    follow_poses[str(frame_num)] = {
                        'id': follow_pose['id'],
                        'bbox': follow_pose['bbox'],
                        'confidence': follow_pose['confidence'],
                        'keypoints': follow_pose['keypoints']
                    }
        
        # Save assignments
        self.save_json(lead_poses, self.lead_file)
        self.save_json(follow_poses, self.follow_file)

    def _parse_role_files(self, role_dir):
        """Parse role directory filenames to get frame/track assignments"""
        assignments = {}
        for filename in os.listdir(role_dir):
            if filename.endswith('.jpg'):
                frame_num, track_id = map(
                    int,
                    filename.split('.')[0].split('-')
                )
                assignments[frame_num] = track_id
        return assignments

    def interpolate_missing_assignments(self):
        """Step 6: Interpolate missing role assignments"""
        print("Interpolating missing assignments...")
        
        lead_poses = self.load_json(self.lead_file)
        follow_poses = self.load_json(self.follow_file)
        
        # Convert to frame numbers
        lead_frames = set(map(int, lead_poses.keys()))
        follow_frames = set(map(int, follow_poses.keys()))
        
        # Get all frame numbers
        all_frames = sorted(set(self.detections.keys()))
        
        # Interpolate lead poses
        self._interpolate_role_poses(
            lead_poses, lead_frames, all_frames, is_lead=True
        )
        
        # Interpolate follow poses
        self._interpolate_role_poses(
            follow_poses, follow_frames, all_frames, is_lead=False
        )
        
        # Save updated assignments
        self.save_json(lead_poses, self.lead_file)
        self.save_json(follow_poses, self.follow_file)

    def _interpolate_role_poses(self, role_poses, role_frames, all_frames, is_lead):
        """Helper to interpolate missing poses for a role"""
        for frame_num in all_frames:
            if frame_num not in role_frames:
                # Find nearest frames with assignments
                prev_frame = max((f for f in role_frames if f < frame_num), default=None)
                next_frame = min((f for f in role_frames if f > frame_num), default=None)
                
                if prev_frame is not None:
                    # Use track ID from previous frame
                    prev_id = role_poses[str(prev_frame)]['id']
                    detections_in_frame = self.detections.get(frame_num, [])
                    matching_pose = next(
                        (d for d in detections_in_frame if d['id'] == prev_id),
                        None
                    )
                    
                    if matching_pose:
                        role_poses[str(frame_num)] = {
                            'id': matching_pose['id'],
                            'bbox': matching_pose['bbox'],
                            'confidence': matching_pose['confidence'],
                            'keypoints': matching_pose['keypoints']
                        }

    #region UTILITY

    @staticmethod
    def load_json(json_path):
        with open(json_path, 'r') as f:
            detections = json.load(f)

        # Convert all frame keys to integers
        return {int(frame): data for frame, data in detections.items()}

    def save_json(self, data, file_path):
        with open(file_path, 'w') as f:
            json.dump(self.numpy_to_python(data), f, indent=2)

    def numpy_to_python(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self.numpy_to_python(obj.tolist())
        elif isinstance(obj, list):
            return [self.numpy_to_python(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self.numpy_to_python(value) for key, value in obj.items()}
        else:
            return obj

    #endregion

