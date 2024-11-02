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

        # Add new path for analysis cache
        self.analysis_cache_file = os.path.join(output_dir, 'face_analysis.json')
        
        # Remove folder creation for lead/follow dirs since we won't use them
        for dir_path in [self.faces_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def process_video(self):
        # Check if faces directory already has crops
        existing_faces = os.listdir(self.faces_dir)
        if not existing_faces:
            self.extract_face_crops()
        else:
            print(f"Found {len(existing_faces)} existing face crops, skipping extraction...")
        
        self.analyze_faces()
        self.create_role_assignments()
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

    def analyze_faces(self):
        """Analyze each face for gender and race, cache results"""
        # Check for existing analysis cache
        if os.path.exists(self.analysis_cache_file):
            print("Loading cached face analysis results...")
            with open(self.analysis_cache_file, 'r') as f:
                self.face_analysis = json.load(f)
            
            # Print statistics from cache
            male_count = sum(1 for data in self.face_analysis.values() 
                            if data['dominant_gender'] == 'Man')
            female_count = sum(1 for data in self.face_analysis.values() 
                              if data['dominant_gender'] == 'Woman')
            print(f"Loaded {male_count} male and {female_count} female cached analyses")
            return

        print("Analyzing faces for gender and race...")
        face_files = os.listdir(self.faces_dir)
        
        if not face_files:
            raise Exception("No face crops found!")
        
        self.face_analysis = {}
        
        for face_file in tqdm.tqdm(face_files):
            try:
                img_path = os.path.join(self.faces_dir, face_file)
                result = DeepFace.analyze(
                    img_path=img_path,
                    actions=['gender', 'race'],
                    enforce_detection=False
                )
                
                if isinstance(result, list):
                    result = result[0]
                
                # Parse frame number and track id from filename
                frame_num, track_id = map(
                    int,
                    face_file.split('.')[0].split('-')
                )
                
                # Store analysis results
                self.face_analysis[face_file] = {
                    'frame_num': frame_num,
                    'track_id': track_id,
                    'dominant_gender': result['dominant_gender'],
                    'gender_confidence': result['gender'][result['dominant_gender']],
                    'dominant_race': result['dominant_race'],
                    'race_confidence': result['race'][result['dominant_race']]
                }
                    
            except Exception as e:
                print(f"Error analyzing {face_file}: {str(e)}")
                continue
        
        # Save analysis cache
        with open(self.analysis_cache_file, 'w') as f:
            json.dump(self.face_analysis, f, indent=2)
        
        # Print statistics
        male_count = sum(1 for data in self.face_analysis.values() 
                        if data['dominant_gender'] == 'Man')
        female_count = sum(1 for data in self.face_analysis.values() 
                          if data['dominant_gender'] == 'Woman')
        print(f"Analyzed {male_count} male and {female_count} female faces")
        
        # Print race statistics
        race_counts = defaultdict(int)
        for data in self.face_analysis.values():
            race_counts[data['dominant_race']] += 1
        print("\nRace distribution:")
        for race, count in race_counts.items():
            print(f"{race}: {count}")

    def create_role_assignments(self):
        """Create lead and follow assignments using multi-factor analysis"""
        print("Creating role assignments with multi-factor analysis...")
        
        # First pass: Analyze tracks for gender and race consistency
        track_analysis = self.analyze_tracks_demographics()
        
        # Second pass: Create frame-by-frame assignments
        lead_poses, follow_poses = self.assign_roles_over_time(track_analysis)
        
        # Save assignments
        self.save_json(lead_poses, self.lead_file)
        self.save_json(follow_poses, self.follow_file)

    def analyze_tracks_demographics(self):
        """Analyze demographic consistency for each track with confidence anchors"""
        track_data = defaultdict(lambda: {
            'frames': [],
            'male_votes': 0,
            'female_votes': 0,
            'male_confidence': 0,
            'female_confidence': 0,
            'race_votes': defaultdict(float),
            'positions': [],  # (frame_num, x, y) tuples
            'high_confidence_points': []  # Will store anchor points
        })
        
        # First pass: Collect all data and identify high confidence points
        for file_name, analysis in self.face_analysis.items():
            track_id = analysis['track_id']
            frame_num = analysis['frame_num']
            
            detection = next(
                (d for d in self.detections.get(frame_num, []) 
                 if d['id'] == track_id),
                None
            )
            
            if detection:
                bbox = detection['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                
                # Store basic track data
                track_data[track_id]['positions'].append((frame_num, center_x, center_y))
                track_data[track_id]['frames'].append(frame_num)
                
                # Add gender vote
                if analysis['dominant_gender'] == 'Man':
                    track_data[track_id]['male_votes'] += 1
                    track_data[track_id]['male_confidence'] += analysis['gender_confidence']
                else:
                    track_data[track_id]['female_votes'] += 1
                    track_data[track_id]['female_confidence'] += analysis['gender_confidence']
                
                # Add race vote
                race = analysis['dominant_race']
                confidence = analysis['race_confidence']
                simplified_race = 'dark' if race in ['indian', 'black'] else 'light'
                track_data[track_id]['race_votes'][simplified_race] += confidence
                
                # Check if this is a high confidence point
                gender_conf = analysis['gender_confidence']
                race_conf = analysis['race_confidence']
                pose_conf = detection['confidence']
                
                # Define criteria for high confidence points
                if (gender_conf > 0.9 and race_conf > 0.8 and pose_conf > 0.8):
                    track_data[track_id]['high_confidence_points'].append({
                        'frame': frame_num,
                        'position': (center_x, center_y),
                        'gender': analysis['dominant_gender'],
                        'race': simplified_race,
                        'confidence': (gender_conf + race_conf + pose_conf) / 3
                    })
        
        # Second pass: Analyze tracks and create bridges
        track_analysis = {}
        for track_id, data in track_data.items():
            if not data['frames']:
                continue
                
            # Sort frames and high confidence points
            data['frames'].sort()
            data['high_confidence_points'].sort(key=lambda x: x['frame'])
            
            # Calculate basic demographics
            male_score = data['male_confidence'] / max(1, data['male_votes'])
            female_score = data['female_confidence'] / max(1, data['female_votes'])
            
            race_scores = data['race_votes']
            dominant_race = max(race_scores.items(), key=lambda x: x[1])[0] if race_scores else None
            
            # Analyze track segments
            segments = self._analyze_track_segments(data)
            
            track_analysis[track_id] = {
                'frames': data['frames'],
                'gender_score': {'male': male_score, 'female': female_score},
                'dominant_race': dominant_race,
                'race_confidence': max(race_scores.values()) if race_scores else 0,
                'segments': segments,
                'high_confidence_points': data['high_confidence_points']
            }
        
        return track_analysis

    def _analyze_track_segments(self, track_data):
        """Analyze track segments between high confidence points"""
        segments = []
        positions = sorted(track_data['positions'], key=lambda x: x[0])
        high_conf_points = track_data['high_confidence_points']
        
        if not high_conf_points:
            return []
        
        # Create segments between high confidence points
        for i in range(len(high_conf_points) - 1):
            start_point = high_conf_points[i]
            end_point = high_conf_points[i + 1]
            
            # Get positions between these points
            segment_positions = [
                pos for pos in positions 
                if start_point['frame'] <= pos[0] <= end_point['frame']
            ]
            
            # Check segment validity
            is_valid = self._validate_segment(
                segment_positions,
                start_point,
                end_point
            )
            
            segments.append({
                'start_frame': start_point['frame'],
                'end_frame': end_point['frame'],
                'is_valid': is_valid,
                'confidence': (start_point['confidence'] + end_point['confidence']) / 2
            })
        
        return segments

    def _validate_segment(self, positions, start_point, end_point):
        """Validate a track segment between two high confidence points"""
        if len(positions) < 2:
            return False
        
        # Check temporal consistency
        frame_gaps = [
            positions[i+1][0] - positions[i][0] 
            for i in range(len(positions)-1)
        ]
        if max(frame_gaps) > 5:  # Allow gaps of up to 5 frames
            return False
        
        # Check spatial consistency
        max_allowed_speed = self.frame_width * 0.1  # 10% of frame width per frame
        for i in range(len(positions)-1):
            dx = positions[i+1][1] - positions[i][1]
            dy = positions[i+1][2] - positions[i][2]
            distance = (dx*dx + dy*dy)**0.5
            frames = positions[i+1][0] - positions[i][0]
            if frames > 0 and distance/frames > max_allowed_speed:
                return False
        
        return True

    def assign_roles_over_time(self, track_analysis):
        """Assign lead and follow roles using track segments and confidence points"""
        lead_poses = {}
        follow_poses = {}
        
        # Group high confidence points by frame
        confidence_points = defaultdict(list)
        for track_id, analysis in track_analysis.items():
            for point in analysis['high_confidence_points']:
                confidence_points[point['frame']].append({
                    'track_id': track_id,
                    'point': point
                })
        
        # First pass: Assign roles at high confidence points
        assigned_tracks = {'lead': set(), 'follow': set()}
        
        for frame_num in sorted(confidence_points.keys()):
            points = confidence_points[frame_num]
            if len(points) >= 2:
                # Score points for lead/follow roles
                lead_candidates = []
                follow_candidates = []
                
                for point_data in points:
                    track_id = point_data['track_id']
                    point = point_data['point']
                    analysis = track_analysis[track_id]
                    
                    male_score = analysis['gender_score']['male']
                    female_score = analysis['gender_score']['female']
                    
                    lead_score = male_score * point['confidence']
                    follow_score = female_score * point['confidence']
                    
                    lead_candidates.append((lead_score, track_id, point))
                    follow_candidates.append((follow_score, track_id, point))
                
                # Assign roles at high confidence points
                self._assign_roles_at_point(
                    frame_num, 
                    lead_candidates, 
                    follow_candidates, 
                    lead_poses, 
                    follow_poses,
                    assigned_tracks
                )
        
        # Second pass: Fill gaps between high confidence points
        all_frames = sorted(set(self.detections.keys()))
        
        for track_id, analysis in track_analysis.items():
            for segment in analysis['segments']:
                if segment['is_valid']:
                    start_frame = segment['start_frame']
                    end_frame = segment['end_frame']
                    
                    # Fill gaps if this track is assigned a role
                    if track_id in assigned_tracks['lead']:
                        self._fill_track_gap(
                            track_id, start_frame, end_frame, lead_poses
                        )
                    elif track_id in assigned_tracks['follow']:
                        self._fill_track_gap(
                            track_id, start_frame, end_frame, follow_poses
                        )
        
        return lead_poses, follow_poses

    def _assign_roles_at_point(self, frame_num, lead_candidates, follow_candidates, 
                              lead_poses, follow_poses, assigned_tracks):
        """Assign roles at a high confidence point"""
        lead_candidates.sort(reverse=True)
        follow_candidates.sort(reverse=True)
        
        # Try to assign lead
        for lead_score, lead_id, point in lead_candidates:
            detection = next(
                (d for d in self.detections.get(frame_num, []) 
                 if d['id'] == lead_id),
                None
            )
            if detection:
                lead_poses[str(frame_num)] = {
                    'id': detection['id'],
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'keypoints': detection['keypoints'],
                    'score': float(lead_score)
                }
                assigned_tracks['lead'].add(lead_id)
                break
        
        # Try to assign follow
        for follow_score, follow_id, point in follow_candidates:
            if follow_id != lead_id:  # Ensure different from lead
                detection = next(
                    (d for d in self.detections.get(frame_num, []) 
                     if d['id'] == follow_id),
                    None
                )
                if detection:
                    follow_poses[str(frame_num)] = {
                        'id': detection['id'],
                        'bbox': detection['bbox'],
                        'confidence': detection['confidence'],
                        'keypoints': detection['keypoints'],
                        'score': float(follow_score)
                    }
                    assigned_tracks['follow'].add(follow_id)
                    break

    def _fill_track_gap(self, track_id, start_frame, end_frame, poses_dict):
        """Fill gaps in tracking between start and end frames"""
        for frame_num in range(start_frame, end_frame + 1):
            if str(frame_num) not in poses_dict:
                detection = next(
                    (d for d in self.detections.get(frame_num, []) 
                     if d['id'] == track_id),
                    None
                )
                if detection:
                    poses_dict[str(frame_num)] = {
                        'id': detection['id'],
                        'bbox': detection['bbox'],
                        'confidence': detection['confidence'],
                        'keypoints': detection['keypoints'],
                        'interpolated': True
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

