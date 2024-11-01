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
        
        self.cluster_faces()
        self.analyze_gender_and_assign_roles()
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

    def cluster_faces(self):
        """Steps 2-3: Cluster faces using DeepFace.find"""
        print("Clustering faces...")
        face_files = os.listdir(self.faces_dir)
        
        if not face_files:
            raise Exception("No face crops found!")
            
        # Take a random sample of faces to analyze
        sample_size = min(20, len(face_files))
        sample_faces = random.sample(face_files, sample_size)
        
        clusters = defaultdict(list)
        processed_files = set()
        
        for face_file in tqdm.tqdm(sample_faces):
            if face_file in processed_files:
                continue
                
            try:
                img_path = os.path.join(self.faces_dir, face_file)
                dfs = DeepFace.find(
                    img_path=img_path,
                    db_path=self.faces_dir,
                    enforce_detection=False
                )
                
                if dfs[0].empty:
                    continue
                    
                # Create new cluster
                cluster_id = len(clusters)
                similar_faces = dfs[0]['identity'].tolist()
                
                # Add similar faces to cluster
                for face_path in similar_faces:
                    face_name = os.path.basename(face_path)
                    clusters[cluster_id].append(face_name)
                    processed_files.add(face_name)
                    
            except Exception as e:
                print(f"Error processing {face_file}: {str(e)}")
                continue
        
        # Keep the two largest clusters
        sorted_clusters = sorted(
            clusters.items(), 
            key=lambda x: len(x[1]), 
            reverse=True
        )[:2]
        
        # Copy files to person directories
        for idx, (cluster_id, files) in enumerate(sorted_clusters):
            person_dir = os.path.join(self.person_dirs, f"person_{idx+1}")
            os.makedirs(person_dir, exist_ok=True)
            
            for file in files:
                src = os.path.join(self.faces_dir, file)
                dst = os.path.join(person_dir, file)
                shutil.copy2(src, dst)

    def analyze_gender_and_assign_roles(self):
        """Step 4: Analyze gender and assign lead/follow roles"""
        print("Analyzing gender and assigning roles...")
        
        person_dirs = [d for d in os.listdir(self.person_dirs) 
                      if d.startswith('person_')]
        
        gender_counts = {}
        
        for person_dir in person_dirs:
            dir_path = os.path.join(self.person_dirs, person_dir)
            files = os.listdir(dir_path)
            
            # Sample up to 10 images for gender analysis
            sample_size = min(10, len(files))
            sample_files = random.sample(files, sample_size)
            
            male_count = 0
            female_count = 0
            
            for file in sample_files:
                try:
                    img_path = os.path.join(dir_path, file)
                    result = DeepFace.analyze(
                        img_path=img_path,
                        actions=['gender'],
                        enforce_detection=False
                    )
                    
                    if isinstance(result, list):
                        result = result[0]
                        
                    if result['gender'] == 'Man':
                        male_count += 1
                    else:
                        female_count += 1
                        
                except Exception as e:
                    print(f"Error analyzing {file}: {str(e)}")
                    continue
            
            gender_counts[person_dir] = {
                'male': male_count,
                'female': female_count
            }
        
        # Assign roles based on gender counts
        sorted_dirs = sorted(
            gender_counts.items(),
            key=lambda x: x[1]['male'],
            reverse=True
        )
        
        if len(sorted_dirs) >= 2:
            lead_dir = sorted_dirs[0][0]
            follow_dir = sorted_dirs[1][0]
            
            # Copy files to role-specific directories
            self._copy_files_to_role_dir(lead_dir, self.lead_faces_dir)
            self._copy_files_to_role_dir(follow_dir, self.follow_faces_dir)
            
            # Create role assignments
            self._create_role_assignments()

    def _copy_files_to_role_dir(self, person_dir, role_dir):
        """Helper to copy files from person directory to role directory"""
        src_dir = os.path.join(self.person_dirs, person_dir)
        for file in os.listdir(src_dir):
            shutil.copy2(
                os.path.join(src_dir, file),
                os.path.join(role_dir, file)
            )

    def _create_role_assignments(self):
        """Create lead and follow assignments based on face analysis"""
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

