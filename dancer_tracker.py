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
        """Analyze demographic consistency with emphasis on track stability"""
        track_data = defaultdict(lambda: {
            'frames': [],
            'male_votes': 0,
            'female_votes': 0,
            'male_confidence': 0,
            'female_confidence': 0,
            'race_votes': defaultdict(float),
            'positions': [],  # (frame_num, x, y) tuples
            'high_confidence_points': [],  # Will store anchor points
            'stable_segments': []  # Will store periods of stable tracking
        })
        
        # First pass: Collect all data and identify stable segments
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
                
                # Add demographic votes with reduced weight
                if analysis['dominant_gender'] == 'Man':
                    track_data[track_id]['male_votes'] += 1
                    track_data[track_id]['male_confidence'] += analysis['gender_confidence']
                else:
                    track_data[track_id]['female_votes'] += 1
                    track_data[track_id]['female_confidence'] += analysis['gender_confidence']
                
                race = analysis['dominant_race']
                confidence = analysis['race_confidence']
                simplified_race = 'dark' if race in ['indian', 'black'] else 'light'
                track_data[track_id]['race_votes'][simplified_race] += confidence
        
        # Identify stable tracking segments
        for track_id, data in track_data.items():
            positions = sorted(data['positions'], key=lambda x: x[0])
            stable_segments = self._find_stable_segments(positions)
            data['stable_segments'] = stable_segments
            
            # Only consider high confidence points within stable segments
            for segment in stable_segments:
                segment_frames = range(segment['start_frame'], segment['end_frame'] + 1)
                for frame_num in segment_frames:
                    analysis = next(
                        (a for a in self.face_analysis.values()
                         if a['frame_num'] == frame_num and a['track_id'] == track_id),
                        None
                    )
                    if analysis and analysis['gender_confidence'] > 0.9:
                        detection = next(
                            (d for d in self.detections.get(frame_num, [])
                             if d['id'] == track_id),
                            None
                        )
                        if detection:
                            data['high_confidence_points'].append({
                                'frame': frame_num,
                                'position': (center_x, center_y),
                                'gender': analysis['dominant_gender'],
                                'race': simplified_race,
                                'confidence': detection['confidence']
                            })
        
        return self._analyze_track_stability(track_data)

    def _find_stable_segments(self, positions):
        """Identify segments of stable tracking based on motion consistency"""
        stable_segments = []
        current_segment = None
        max_speed = self.frame_width * 0.1  # 10% of frame width per frame
        
        for i in range(len(positions) - 1):
            curr_frame, curr_x, curr_y = positions[i]
            next_frame, next_x, next_y = positions[i + 1]
            
            # Calculate motion
            frame_diff = next_frame - curr_frame
            if frame_diff == 0:
                continue
                
            distance = ((next_x - curr_x)**2 + (next_y - curr_y)**2)**0.5
            speed = distance / frame_diff
            
            # Check if motion is stable
            if speed <= max_speed and frame_diff <= 3:  # Allow small gaps
                if current_segment is None:
                    current_segment = {
                        'start_frame': curr_frame,
                        'end_frame': next_frame,
                        'positions': [positions[i]]
                    }
                else:
                    current_segment['end_frame'] = next_frame
                    current_segment['positions'].append(positions[i])
            else:
                if current_segment is not None:
                    if len(current_segment['positions']) > 5:  # Minimum segment length
                        stable_segments.append(current_segment)
                    current_segment = None
        
        # Add final segment if it exists
        if current_segment is not None and len(current_segment['positions']) > 5:
            stable_segments.append(current_segment)
        
        return stable_segments

    def _analyze_track_stability(self, track_data):
        """Analyze tracks with emphasis on stability and proximity"""
        track_analysis = {}
        
        # First pass: Calculate basic stability metrics
        for track_id, data in track_data.items():
            if not data['frames']:
                continue
            
            # Calculate demographic consensus
            total_votes = data['male_votes'] + data['female_votes']
            if total_votes > 0:
                male_ratio = data['male_votes'] / total_votes
                female_ratio = data['female_votes'] / total_votes
                gender_consensus = max(male_ratio, female_ratio)
            else:
                gender_consensus = 0
            
            # Calculate race consensus
            race_votes = data['race_votes']
            total_race_votes = sum(race_votes.values())
            race_consensus = max(race_votes.values()) / total_race_votes if total_race_votes > 0 else 0
            
            # Calculate tracking stability score
            stability_score = sum(
                segment['end_frame'] - segment['start_frame'] 
                for segment in data['stable_segments']
            ) / len(data['frames']) if data['frames'] else 0
            
            track_analysis[track_id] = {
                'frames': sorted(data['frames']),
                'stable_segments': data['stable_segments'],
                'gender_consensus': gender_consensus,
                'dominant_gender': 'Man' if data['male_votes'] > data['female_votes'] else 'Woman',
                'race_consensus': race_consensus,
                'dominant_race': max(data['race_votes'].items(), key=lambda x: x[1])[0] if data['race_votes'] else None,
                'stability_score': stability_score,
                'positions': sorted(data['positions'], key=lambda x: x[0])
            }
        
        return track_analysis

    def assign_roles_over_time(self, track_analysis):
        """Assign roles with emphasis on track stability and ensuring role coverage"""
        lead_poses = {}
        follow_poses = {}
        
        # Sort tracks by stability score
        sorted_tracks = sorted(
            track_analysis.items(),
            key=lambda x: x[1]['stability_score'],
            reverse=True
        )
        
        # Initialize role assignments
        current_lead = None
        current_follow = None
        
        # Process frames in order
        all_frames = sorted(set(self.detections.keys()))
        
        for frame_num in all_frames:
            detections_in_frame = self.detections.get(frame_num, [])
            
            # Get active tracks in this frame
            active_tracks = []
            for track_id, analysis in track_analysis.items():
                if frame_num in analysis['frames']:
                    detection = next(
                        (d for d in detections_in_frame if d['id'] == track_id),
                        None
                    )
                    if detection:
                        active_tracks.append((track_id, detection, analysis))
            
            if not active_tracks:
                continue
                
            # Check for close proximity situations
            proximity_warning = self._check_proximity(active_tracks)
            
            # Update role assignments
            if proximity_warning:
                # Use more careful assignment when poses are close
                self._assign_roles_proximity(
                    frame_num,
                    active_tracks,
                    lead_poses,
                    follow_poses,
                    current_lead,
                    current_follow
                )
            else:
                # Use stable tracking when poses are far apart
                self._assign_roles_stable(
                    frame_num,
                    active_tracks,
                    lead_poses,
                    follow_poses,
                    current_lead,
                    current_follow
                )
            
            # Ensure roles are assigned if we have detections
            self._enforce_role_coverage(
                frame_num,
                active_tracks,
                lead_poses,
                follow_poses,
                current_lead,
                current_follow
            )
            
            # Update current assignments
            if str(frame_num) in lead_poses:
                current_lead = lead_poses[str(frame_num)]['id']
            if str(frame_num) in follow_poses:
                current_follow = follow_poses[str(frame_num)]['id']
        
        return lead_poses, follow_poses

    def _enforce_role_coverage(self, frame_num, active_tracks, lead_poses, follow_poses,
                               current_lead, current_follow):
        """Ensure that roles are assigned when poses are available"""
        frame_str = str(frame_num)
        
        # If we have exactly one detection, assign it to maintain the most recent role
        if len(active_tracks) == 1:
            track_id, detection, analysis = active_tracks[0]
            
            # If this track was previously assigned a role, maintain it
            if track_id == current_lead:
                if frame_str not in lead_poses:
                    lead_poses[frame_str] = {
                        'id': detection['id'],
                        'bbox': detection['bbox'],
                        'confidence': detection['confidence'],
                        'keypoints': detection['keypoints'],
                        'single_detection': True
                    }
            elif track_id == current_follow:
                if frame_str not in follow_poses:
                    follow_poses[frame_str] = {
                        'id': detection['id'],
                        'bbox': detection['bbox'],
                        'confidence': detection['confidence'],
                        'keypoints': detection['keypoints'],
                        'single_detection': True
                    }
            else:
                # New track, assign based on gender if confident, otherwise prefer lead
                if (analysis['gender_consensus'] > 0.8 and 
                    analysis['dominant_gender'] == 'Woman'):
                    follow_poses[frame_str] = {
                        'id': detection['id'],
                        'bbox': detection['bbox'],
                        'confidence': detection['confidence'],
                        'keypoints': detection['keypoints'],
                        'single_detection': True
                    }
                else:
                    lead_poses[frame_str] = {
                        'id': detection['id'],
                        'bbox': detection['bbox'],
                        'confidence': detection['confidence'],
                        'keypoints': detection['keypoints'],
                        'single_detection': True
                    }
        
        # If we have two or more detections, ensure both roles are assigned
        elif len(active_tracks) >= 2:
            if frame_str not in lead_poses or frame_str not in follow_poses:
                # Score all tracks for both roles
                scores = []
                for track_id, detection, analysis in active_tracks:
                    lead_score = (
                        (analysis['stability_score'] * 2) +
                        (1.0 if analysis['dominant_gender'] == 'Man' else 0) +
                        (0.5 if track_id == current_lead else 0)
                    )
                    follow_score = (
                        (analysis['stability_score'] * 2) +
                        (1.0 if analysis['dominant_gender'] == 'Woman' else 0) +
                        (0.5 if track_id == current_follow else 0)
                    )
                    scores.append((track_id, detection, analysis, lead_score, follow_score))
                
                # Sort by total score
                scores.sort(key=lambda x: x[3] + x[4], reverse=True)
                
                # Assign the two highest scoring tracks to roles
                if frame_str not in lead_poses and frame_str not in follow_poses:
                    # Assign best two tracks based on their relative scores
                    track1_id, det1, _, lead1_score, follow1_score = scores[0]
                    track2_id, det2, _, lead2_score, follow2_score = scores[1]
                    
                    if lead1_score > follow1_score:
                        # First track is lead
                        lead_poses[frame_str] = {
                            'id': det1['id'],
                            'bbox': det1['bbox'],
                            'confidence': det1['confidence'],
                            'keypoints': det1['keypoints'],
                            'enforced': True
                        }
                        follow_poses[frame_str] = {
                            'id': det2['id'],
                            'bbox': det2['bbox'],
                            'confidence': det2['confidence'],
                            'keypoints': det2['keypoints'],
                            'enforced': True
                        }
                    else:
                        # First track is follow
                        follow_poses[frame_str] = {
                            'id': det1['id'],
                            'bbox': det1['bbox'],
                            'confidence': det1['confidence'],
                            'keypoints': det1['keypoints'],
                            'enforced': True
                        }
                        lead_poses[frame_str] = {
                            'id': det2['id'],
                            'bbox': det2['bbox'],
                            'confidence': det2['confidence'],
                            'keypoints': det2['keypoints'],
                            'enforced': True
                        }
                elif frame_str not in lead_poses:
                    # Find best unassigned track for lead
                    follow_id = follow_poses[frame_str]['id']
                    for track_id, detection, _, lead_score, _ in scores:
                        if track_id != follow_id:
                            lead_poses[frame_str] = {
                                'id': detection['id'],
                                'bbox': detection['bbox'],
                                'confidence': detection['confidence'],
                                'keypoints': detection['keypoints'],
                                'enforced': True
                            }
                            break
                elif frame_str not in follow_poses:
                    # Find best unassigned track for follow
                    lead_id = lead_poses[frame_str]['id']
                    for track_id, detection, _, _, follow_score in scores:
                        if track_id != lead_id:
                            follow_poses[frame_str] = {
                                'id': detection['id'],
                                'bbox': detection['bbox'],
                                'confidence': detection['confidence'],
                                'keypoints': detection['keypoints'],
                                'enforced': True
                            }
                            break

    def _check_proximity(self, active_tracks):
        """Check if any two poses are in close proximity"""
        if len(active_tracks) < 2:
            return False
            
        for i in range(len(active_tracks)):
            for j in range(i + 1, len(active_tracks)):
                _, det1, _ = active_tracks[i]
                _, det2, _ = active_tracks[j]
                
                # Calculate center points
                center1 = ((det1['bbox'][0] + det1['bbox'][2])/2,
                          (det1['bbox'][1] + det1['bbox'][3])/2)
                center2 = ((det2['bbox'][0] + det2['bbox'][2])/2,
                          (det2['bbox'][1] + det2['bbox'][3])/2)
                
                # Calculate distance
                distance = ((center1[0] - center2[0])**2 +
                           (center1[1] - center2[1])**2)**0.5
                
                # Check if distance is less than average person width
                avg_width = (det1['bbox'][2] - det1['bbox'][0] +
                            det2['bbox'][2] - det2['bbox'][0]) / 2
                
                if distance < avg_width * 1.5:  # Proximity threshold
                    return True
        
        return False

    def _assign_roles_stable(self, frame_num, active_tracks, lead_poses, follow_poses,
                            current_lead, current_follow):
        """Assign roles with high inertia when tracks are stable"""
        for track_id, detection, analysis in active_tracks:
            # Maintain current assignments if track is stable
            if track_id == current_lead:
                lead_poses[str(frame_num)] = {
                    'id': detection['id'],
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'keypoints': detection['keypoints'],
                    'stable': True
                }
            elif track_id == current_follow:
                follow_poses[str(frame_num)] = {
                    'id': detection['id'],
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'keypoints': detection['keypoints'],
                    'stable': True
                }

    def _assign_roles_proximity(self, frame_num, active_tracks, lead_poses, follow_poses,
                                current_lead, current_follow):
        """Carefully assign roles when poses are in close proximity"""
        # Score each track based on multiple factors
        scores = []
        for track_id, detection, analysis in active_tracks:
            score = 0
            
            # Add stability score
            score += analysis['stability_score'] * 2
            
            # Add demographic consensus score
            if analysis['dominant_gender'] == 'Man':
                score += analysis['gender_consensus']
            
            # Add continuity bonus
            if track_id == current_lead:
                score += 0.5
            elif track_id == current_follow:
                score -= 0.5
            
            scores.append((score, track_id, detection, analysis))
        
        # Sort by score
        scores.sort(reverse=True)
        
        # Assign roles
        if len(scores) >= 2:
            lead_score, lead_id, lead_detection, _ = scores[0]
            follow_score, follow_id, follow_detection, _ = scores[1]
            
            lead_poses[str(frame_num)] = {
                'id': lead_detection['id'],
                'bbox': lead_detection['bbox'],
                'confidence': lead_detection['confidence'],
                'keypoints': lead_detection['keypoints'],
                'score': float(lead_score),
                'proximity': True
            }
            
            follow_poses[str(frame_num)] = {
                'id': follow_detection['id'],
                'bbox': follow_detection['bbox'],
                'confidence': follow_detection['confidence'],
                'keypoints': follow_detection['keypoints'],
                'score': float(follow_score),
                'proximity': True
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

