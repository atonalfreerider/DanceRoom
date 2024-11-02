import cv2
import numpy as np
import json
import os
import tqdm
from deepface import DeepFace
from collections import defaultdict

# uses DeepFace to find self-similar faces and gender
class DancerTracker:
    def __init__(self, input_path, output_dir):
        self.input_path = input_path
        self.output_dir = output_dir
        self.depth_dir = os.path.join(output_dir, 'depth')

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

        self.face_analysis = {}

    def process_video(self):
        # Check if we have cached analysis
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
        else:
            self.analyze_video_faces()
        
        self.create_role_assignments()
        print("Lead and follow tracked using DeepFace approach")

    def analyze_video_faces(self):
        """Extract and analyze faces directly from video frames"""
        print("Analyzing faces from video...")
        
        # Open video capture
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise Exception(f"Could not open video file: {self.input_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        min_height_threshold = 0.6 * self.frame_height
        
        for frame_num in tqdm.tqdm(range(total_frames)):
            # Read frame
            ret, frame = cap.read()
            if not ret:
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
                            try:
                                # Analyze face directly from numpy array
                                result = DeepFace.analyze(
                                    img_path=head_img,  # Pass numpy array directly
                                    actions=['gender', 'race'],
                                    enforce_detection=False,
                                    silent=True
                                )
                                
                                if isinstance(result, list):
                                    result = result[0]
                                
                                # Generate a unique key for this face
                                face_key = f"{frame_num:06d}-{detection['id']}"
                                
                                # Store analysis results
                                self.face_analysis[face_key] = {
                                    'frame_num': frame_num,
                                    'track_id': detection['id'],
                                    'dominant_gender': result['dominant_gender'],
                                    'gender_confidence': result['gender'][result['dominant_gender']],
                                    'dominant_race': result['dominant_race'],
                                    'race_confidence': result['race'][result['dominant_race']]
                                }
                                    
                            except Exception as e:
                                print(f"Error analyzing face in frame {frame_num}, "
                                      f"track {detection['id']}: {str(e)}")
                                continue
        
        # Release video capture
        cap.release()
        
        # Save analysis cache
        with open(self.analysis_cache_file, 'w') as f:
            json.dump(self.face_analysis, f, indent=2)
        
        # Print statistics
        male_count = sum(1 for data in self.face_analysis.values() 
                        if data['dominant_gender'] == 'Man')
        female_count = sum(1 for data in self.face_analysis.values() 
                          if data['dominant_gender'] == 'Woman')
        print(f"\nAnalyzed {male_count} male and {female_count} female faces")
        
        # Print race statistics
        race_counts = defaultdict(int)
        for data in self.face_analysis.values():
            race_counts[data['dominant_race']] += 1
        print("\nRace distribution:")
        for race, count in race_counts.items():
            print(f"{race}: {count}")

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
        # Initialize track_data with explicit types for each field
        track_data = defaultdict(lambda: {
            'frames': [],
            'male_votes': 0,
            'female_votes': 0,
            'male_confidence': 0.0,
            'female_confidence': 0.0,
            'race_votes': defaultdict(float),
            'positions': [],
            'high_confidence_points': [],
            'stable_segments': [],
            'size_ratios': [],
            'size_score': 0.0  # Initialize size_score as float
        })

        simplified_race = 'dark'
        
        # First pass: Collect all data
        for file_name, analysis in self.face_analysis.items():
            track_id = analysis['track_id']
            frame_num = analysis['frame_num']
            
            detection = next(
                (d for d in self.detections.get(frame_num, []) 
                 if d['id'] == track_id),
                None
            )

            if detection:
                # Add size measurement
                size = self._calculate_person_size(detection['keypoints'])
                if size > 0:
                    track_data[track_id]['size_ratios'].append(size)

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
                            bbox = detection['bbox']
                            center_x = (bbox[0] + bbox[2]) / 2
                            center_y = (bbox[1] + bbox[3]) / 2

                            data['high_confidence_points'].append({
                                'frame': frame_num,
                                'position': (center_x, center_y),
                                'gender': analysis['dominant_gender'],
                                'race': simplified_race,
                                'confidence': detection['confidence']
                            })
        
        # Second pass: Calculate relative sizes for each frame with two people
        for frame_num in self.detections:
            detections = self.detections[frame_num]
            if len(detections) == 2:
                det1, det2 = detections
                ratio = self._calculate_relative_size_ratio(
                    det1['keypoints'],
                    det2['keypoints']
                )
                if ratio is not None:
                    track_data[det1['id']]['size_ratios'].append(ratio)
                    track_data[det2['id']]['size_ratios'].append(1/ratio)
        
        # Calculate size scores from ratios
        for track_id, data in track_data.items():
            if data['size_ratios']:
                # Remove outliers (ratios outside 2 standard deviations)
                ratios = np.array(data['size_ratios'])
                mean_ratio = np.mean(ratios)
                std_ratio = np.std(ratios)
                filtered_ratios = ratios[np.abs(ratios - mean_ratio) <= 2 * std_ratio]
                
                # Convert median ratio to a 0-1 score where >1 means larger
                median_ratio = np.median(filtered_ratios) if len(filtered_ratios) > 0 else mean_ratio
                data['size_score'] = float(1 / (1 + np.exp(-2 * (median_ratio - 1))))  # Explicit float conversion
            else:
                data['size_score'] = 0.5  # Neutral score if no ratios available
        
        # Modify track analysis to include size
        track_analysis = {}
        for track_id, data in track_data.items():
            if not data['frames']:
                continue
            
            # Calculate average size when enough measurements exist
            size_measurements = data['size_ratios']
            if len(size_measurements) > 5:  # Require minimum measurements
                # Remove outliers (measurements outside 2 standard deviations)
                mean_size = np.mean(size_measurements)
                std_size = np.std(size_measurements)
                filtered_sizes = [
                    s for s in size_measurements 
                    if abs(s - mean_size) <= 2 * std_size
                ]
                avg_size = np.mean(filtered_sizes) if filtered_sizes else mean_size
            else:
                avg_size = np.mean(size_measurements) if size_measurements else 0
            
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
                'positions': sorted(data['positions'], key=lambda x: x[0]),
                'average_size': avg_size  # Add size to analysis
            }
        
        # Normalize size scores across all tracks
        if track_analysis:
            size_scores = [t['average_size'] for t in track_analysis.values() if t['average_size'] > 0]
            if size_scores:
                min_size = min(size_scores)
                max_size = max(size_scores)
                size_range = max_size - min_size
                if size_range > 0:
                    for track_id in track_analysis:
                        if track_analysis[track_id]['average_size'] > 0:
                            track_analysis[track_id]['size_score'] = (
                                (track_analysis[track_id]['average_size'] - min_size) / size_range
                            )
                        else:
                            track_analysis[track_id]['size_score'] = 0
        
        return track_analysis

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

    def assign_roles_over_time(self, track_analysis):
        """Assign roles with emphasis on track stability and ensuring role coverage"""
        lead_poses = {}
        follow_poses = {}

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

    @staticmethod
    def _enforce_role_coverage(frame_num, active_tracks, lead_poses, follow_poses,
                               current_lead, current_follow):
        """Ensure that roles are assigned when poses are available"""
        frame_str = str(frame_num)
        
        if len(active_tracks) >= 2:
            if frame_str not in lead_poses or frame_str not in follow_poses:
                scores = []
                for track_id, detection, analysis in active_tracks:
                    # Calculate size-weighted lead score
                    size_score = analysis.get('size_score', 0.5)
                    size_factor = 6.0 if size_score > 0.6 else (-6.0 if size_score < 0.4 else 0)
                    
                    lead_score = (
                        (analysis['stability_score']) +  # Base stability weight
                        (8.0 if analysis['dominant_gender'] == 'Man' else 0) +  # Increased from 3.0 to 8.0
                        size_factor +  # Heavy size factor
                        (0.25 if track_id == current_lead else 0)  # Reduced continuity bonus
                    )
                    
                    # Inverse scoring for follow role
                    follow_score = (
                        (analysis['stability_score']) +
                        (8.0 if analysis['dominant_gender'] == 'Woman' else 0) +  # Increased from 3.0 to 8.0
                        (-size_factor) +  # Inverse size factor
                        (0.25 if track_id == current_follow else 0)
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

    @staticmethod
    def _check_proximity(active_tracks):
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

    @staticmethod
    def _assign_roles_stable(frame_num, active_tracks, lead_poses, follow_poses,
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

    @staticmethod
    def _assign_roles_proximity(frame_num, active_tracks, lead_poses, follow_poses,
                                current_lead, current_follow):
        """Carefully assign roles when poses are in close proximity"""
        scores = []
        for track_id, detection, analysis in active_tracks:
            score = 0
            
            # Add stability score (reduced relative weight)
            score += analysis['stability_score']
            
            # Add demographic consensus score (heavily increased)
            if analysis['dominant_gender'] == 'Man':
                score += analysis['gender_consensus'] * 8.0  # Increased from 3.0 to 8.0
            
            # Add heavily weighted size score
            size_score = analysis.get('size_score', 0.5)
            if size_score > 0.6:  # If clearly larger
                score += 6.0  # Strong bonus for lead role
            elif size_score < 0.4:  # If clearly smaller
                score -= 6.0  # Strong penalty for lead role
            
            # Add continuity bonus (reduced relative to size and gender)
            if track_id == current_lead:
                score += 0.25
            elif track_id == current_follow:
                score -= 0.25
            
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

    @staticmethod
    def _calculate_person_size(keypoints):
        """Calculate person size based on limb lengths and torso measurements"""
        # Skip points with low confidence or (0,0) coordinates
        valid_points = [
            (x, y) for x, y, conf in keypoints 
            if conf > 0.3 and (x != 0 or y != 0)
        ]
        
        if len(valid_points) < 5:  # Need minimum points for size calculation
            return 0
        
        total_length = 0
        point_count = 0
        
        # Keypoint indices for different body parts
        connections = [
            # Arms
            (5, 7),  # Left upper arm
            (7, 9),  # Left lower arm
            (6, 8),  # Right upper arm
            (8, 10), # Right lower arm
            # Legs
            (11, 13), # Left upper leg
            (13, 15), # Left lower leg
            (12, 14), # Right upper leg
            (14, 16), # Right lower leg
            # Torso measurements
            (5, 11),  # Left torso
            (6, 12),  # Right torso
            (5, 6),   # Shoulders
            (11, 12), # Hips
        ]
        
        # Calculate lengths between connected points
        for start_idx, end_idx in connections:
            if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                keypoints[start_idx][2] > 0.3 and keypoints[end_idx][2] > 0.3 and
                not (keypoints[start_idx][0] == 0 and keypoints[start_idx][1] == 0) and
                not (keypoints[end_idx][0] == 0 and keypoints[end_idx][1] == 0)):
                
                start_point = keypoints[start_idx][:2]
                end_point = keypoints[end_idx][:2]
                
                length = ((end_point[0] - start_point[0])**2 + 
                         (end_point[1] - start_point[1])**2)**0.5
                
                total_length += length
                point_count += 1
        
        return total_length / point_count if point_count > 0 else 0

    @staticmethod
    def _calculate_relative_size_ratio(keypoints1, keypoints2):
        """Calculate size ratio between two people using only torso and legs"""
        # Key measurements to compare
        measurements = [
            (11, 13),  # Left upper leg
            (13, 15),  # Left lower leg
            (12, 14),  # Right upper leg
            (14, 16),  # Right lower leg
            (5, 11),   # Left torso
            (6, 12),   # Right torso
        ]
        
        ratios = []
        
        for start_idx, end_idx in measurements:
            # Check if both people have valid measurements for this segment
            if (start_idx < len(keypoints1) and end_idx < len(keypoints1) and
                start_idx < len(keypoints2) and end_idx < len(keypoints2)):
                
                # Check confidence and validity for person 1
                valid1 = (keypoints1[start_idx][2] > 0.3 and 
                         keypoints1[end_idx][2] > 0.3 and
                         not (keypoints1[start_idx][0] == 0 and keypoints1[start_idx][1] == 0) and
                         not (keypoints1[end_idx][0] == 0 and keypoints1[end_idx][1] == 0))
                
                # Check confidence and validity for person 2
                valid2 = (keypoints2[start_idx][2] > 0.3 and 
                         keypoints2[end_idx][2] > 0.3 and
                         not (keypoints2[start_idx][0] == 0 and keypoints2[start_idx][1] == 0) and
                         not (keypoints2[end_idx][0] == 0 and keypoints2[end_idx][1] == 0))
                
                # Only calculate ratio if both people have valid measurements
                if valid1 and valid2:
                    # Calculate length for person 1
                    length1 = np.sqrt(
                        (keypoints1[end_idx][0] - keypoints1[start_idx][0])**2 +
                        (keypoints1[end_idx][1] - keypoints1[start_idx][1])**2
                    )
                    
                    # Calculate length for person 2
                    length2 = np.sqrt(
                        (keypoints2[end_idx][0] - keypoints2[start_idx][0])**2 +
                        (keypoints2[end_idx][1] - keypoints2[start_idx][1])**2
                    )
                    
                    if length2 > 0:  # Avoid division by zero
                        ratios.append(length1 / length2)
        
        # Return median ratio if we have enough measurements
        if len(ratios) >= 2:  # Require at least 2 valid comparisons
            return np.median(ratios)
        return None

    #endregion

