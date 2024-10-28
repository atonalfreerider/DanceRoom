import cv2
import numpy as np
from ultralytics import YOLO
import json
import os
import tqdm

# uses YOLOv11 to detect gender of figures and track them
class DancerTracker:
    def __init__(self, input_path, output_dir):
        self.input_path = input_path
        self.output_dir = output_dir
        self.depth_dir = os.path.join(output_dir, 'depth')
        self.figure_mask_dir = os.path.join(output_dir, 'figure-masks')

        self.detections_file = os.path.join(output_dir, 'detections.json')
        self.lead_file = os.path.join(output_dir, 'lead.json')
        self.follow_file = os.path.join(output_dir, 'follow.json')

        self.detections = self.load_json(self.detections_file)

        # Initialize YOLO model for gender detection
        self.gender_model = YOLO('/home/john/Desktop/3DPose/DanceRoom/yolo11x-man-woman2.pt')  # You'll need to specify your gender model path

        self.gender_detections_file = os.path.join(output_dir, 'gender-detections.json')
        self.debug_video_path = os.path.join(output_dir, 'gender-debug.mp4')

        # Get input video framerate
        cap = cv2.VideoCapture(input_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        self.gender_confidence_threshold = 0.85  # 85% confidence threshold

    def process_video(self):
        self.track_lead_and_follow()
        print("lead and follow tracked")

    def get_head_bbox(self, keypoints, padding_percent=0.25):
        """Extract head bounding box from keypoints with padding"""
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
        
        # Add padding
        width = x_max - x_min
        height = y_max - y_min
        padding_x = width * padding_percent
        padding_y = height * padding_percent
        
        x_min = max(0, x_min - padding_x)
        y_min = max(0, y_min - padding_y)
        x_max = x_max + padding_x
        y_max = y_max + padding_y
        
        return [int(x_min), int(y_min), int(x_max), int(y_max)]

    def track_lead_and_follow(self):
        frame_count = len(os.listdir(self.figure_mask_dir))
        pbar = tqdm.tqdm(total=frame_count, desc="Tracking poses")

        # Try to load cached gender detections
        gender_detections = {}
        if os.path.exists(self.gender_detections_file):
            print("Loading cached gender detections...")
            with open(self.gender_detections_file, 'r') as f:
                gender_detections = json.load(f)
        
        # First pass: Collect gender votes for each detection
        track_gender_votes = {}
        track_frames = {}
        
        for frame_num in range(frame_count):
            mask_path = os.path.join(self.figure_mask_dir, f"{frame_num:06d}.png")
            if not os.path.exists(mask_path):
                continue
                
            frame_img = cv2.imread(mask_path)
            if frame_img is None:
                continue
            
            frame_height, frame_width = frame_img.shape[:2]
            detections_in_frame = self.detections.get(frame_num, [])
            
            # Initialize frame in gender_detections if not exists
            if str(frame_num) not in gender_detections:
                gender_detections[str(frame_num)] = []

            for detection in detections_in_frame:
                track_id = detection.get('id')
                
                if track_id not in track_frames:
                    track_frames[track_id] = set()
                track_frames[track_id].add(frame_num)
                
                # Check if we have cached gender detection
                cached_gender = next(
                    (g for g in gender_detections.get(str(frame_num), [])
                     if g['id'] == track_id),
                    None
                )
                
                if cached_gender:
                    gender = cached_gender['gender']
                    confidence = cached_gender['confidence']
                else:
                    # Default values
                    gender = 'unknown'
                    confidence = 0
                    
                    # Perform gender detection
                    keypoints = detection.get('keypoints', [])
                    head_bbox = self.get_head_bbox(keypoints)
                    
                    if head_bbox is not None:
                        x1, y1, x2, y2 = head_bbox
                        x1 = max(0, min(x1, frame_width - 1))
                        y1 = max(0, min(y1, frame_height - 1))
                        x2 = max(0, min(x2, frame_width))
                        y2 = max(0, min(y2, frame_height))
                        
                        if x2 > x1 and y2 > y1:
                            try:
                                head_img = frame_img[y1:y2, x1:x2]
                                if head_img.size > 0 and head_img.shape[0] > 0 and head_img.shape[1] > 0:
                                    results = self.gender_model(head_img, verbose=False)
                                    if results and len(results) > 0 and len(results[0].boxes) > 0:
                                        gender_class = int(results[0].boxes[0].cls.cpu().numpy())
                                        confidence = float(results[0].boxes[0].conf.cpu().numpy())
                                        
                                        # Only assign gender if confidence is above threshold
                                        if confidence >= self.gender_confidence_threshold:
                                            gender = 'male' if gender_class == 0 else 'female'
                                        else:
                                            gender = 'unknown'
                                            confidence = 0
                            except Exception as e:
                                print(f"Error processing frame {frame_num}, detection {track_id}: {str(e)}")
                    
                    # Cache the detection
                    gender_detections[str(frame_num)].append({
                        'id': track_id,
                        'gender': gender,
                        'confidence': confidence,
                        'bbox': detection['bbox']
                    })

                detection['gender'] = {'gender': gender, 'confidence': confidence}
                if track_id not in track_gender_votes:
                    track_gender_votes[track_id] = {'male': 0, 'female': 0}
                
                if gender in ['male', 'female']:
                    track_gender_votes[track_id][gender] += confidence

            pbar.update(1)
        
        pbar.close()
        
        # Save gender detections cache
        if not os.path.exists(self.gender_detections_file):
            with open(self.gender_detections_file, 'w') as f:
                json.dump(gender_detections, f, indent=2)

        # Calculate persistence for each track
        track_persistence = {
            track_id: len(frames) 
            for track_id, frames in track_frames.items()
        }

        # Determine primary gender for each track based on weighted votes
        track_genders = {}  # {track_id: 'male'/'female'}
        for track_id, votes in track_gender_votes.items():
            male_votes = votes['male']
            female_votes = votes['female']
            
            # Assign gender based on plurality
            if male_votes > female_votes:
                track_genders[track_id] = 'male'
            else:
                track_genders[track_id] = 'female'

        # Second pass: Frame-by-frame assignment ensuring one lead and one follow
        lead_poses = {}
        follow_poses = {}

        for frame_num in range(frame_count):
            detections_in_frame = self.detections.get(frame_num, [])
            
            # Get active tracks in this frame sorted by persistence
            active_tracks = [
                (d.get('id'), track_persistence[d.get('id')])
                for d in detections_in_frame
                if d.get('id') in track_persistence
            ]
            active_tracks.sort(key=lambda x: x[1], reverse=True)  # Sort by persistence
            
            lead_assigned = False
            follow_assigned = False
            
            # First pass: Try to assign based on voted gender
            for track_id, _ in active_tracks:
                if track_genders[track_id] == 'male' and not lead_assigned:
                    lead_pose = next(d for d in detections_in_frame if d.get('id') == track_id)
                    lead_poses[str(frame_num)] = {
                        'id': lead_pose['id'],
                        'bbox': lead_pose['bbox'],
                        'confidence': lead_pose['confidence'],
                        'keypoints': lead_pose['keypoints']
                    }
                    lead_assigned = True
                elif track_genders[track_id] == 'female' and not follow_assigned:
                    follow_pose = next(d for d in detections_in_frame if d.get('id') == track_id)
                    follow_poses[str(frame_num)] = {
                        'id': follow_pose['id'],
                        'bbox': follow_pose['bbox'],
                        'confidence': follow_pose['confidence'],
                        'keypoints': follow_pose['keypoints']
                    }
                    follow_assigned = True
                
                if lead_assigned and follow_assigned:
                    break
            
            # Second pass: Force assignment if needed
            if not (lead_assigned and follow_assigned) and active_tracks:
                for track_id, _ in active_tracks:
                    if not lead_assigned:
                        lead_pose = next(d for d in detections_in_frame if d.get('id') == track_id)
                        lead_poses[str(frame_num)] = {
                            'id': lead_pose['id'],
                            'bbox': lead_pose['bbox'],
                            'confidence': lead_pose['confidence'],
                            'keypoints': lead_pose['keypoints']
                        }
                        lead_assigned = True
                    elif not follow_assigned:
                        follow_pose = next(d for d in detections_in_frame if d.get('id') == track_id)
                        follow_poses[str(frame_num)] = {
                            'id': follow_pose['id'],
                            'bbox': follow_pose['bbox'],
                            'confidence': follow_pose['confidence'],
                            'keypoints': follow_pose['keypoints']
                        }
                        follow_assigned = True
                    
                    if lead_assigned and follow_assigned:
                        break

        # Save the results
        self.save_json(lead_poses, self.lead_file)
        self.save_json(follow_poses, self.follow_file)

        print(f"Tracked lead and follow poses for {frame_count} frames.")

    def create_debug_video(self):
        """Creates a debug video showing gender detections with colored boxes"""
        frame_count = len(os.listdir(self.figure_mask_dir))
        
        # Load gender detections
        if not os.path.exists(self.gender_detections_file):
            print("No gender detections file found. Run track_lead_and_follow first.")
            return
            
        with open(self.gender_detections_file, 'r') as f:
            gender_detections = json.load(f)
        
        # Get first frame to determine video dimensions
        first_frame = cv2.imread(os.path.join(self.figure_mask_dir, f"{0:06d}.png"))
        if first_frame is None:
            print("Could not read first frame")
            return
            
        height, width = first_frame.shape[:2]
        
        # Initialize video writer with input video's framerate
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.debug_video_path, fourcc, self.fps, (width, height))
        
        pbar = tqdm.tqdm(total=frame_count, desc="Creating debug video")
        
        for frame_num in range(frame_count):
            frame_path = os.path.join(self.figure_mask_dir, f"{frame_num:06d}.png")
            if not os.path.exists(frame_path):
                continue
                
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            
            # Draw detections for this frame
            frame_detections = gender_detections.get(str(frame_num), [])
            for detection in frame_detections:
                gender = detection['gender']
                confidence = detection['confidence']
                bbox = detection['bbox']

                if confidence < 0.85: continue
                
                if gender == 'male':
                    color = (255, 0, 0)  # Blue for men
                elif gender == 'female':
                    color = (255, 0, 255)  # Magenta for women
                else:
                    # Draw gray box for unknown/low confidence detections
                    color = (128, 128, 128)
                
                # Apply alpha based on confidence
                alpha = max(0.3, min(1.0, confidence))
                color = tuple(int(c * alpha) for c in color)
                
                # Draw bbox
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw confidence score and track ID
                text = f"ID:{detection['id']} {confidence:.2f}"
                cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, color, 2)
            
            out.write(frame)
            pbar.update(1)
        
        pbar.close()
        out.release()
        print(f"Debug video saved to {self.debug_video_path}")

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

