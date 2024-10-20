import json
from collections import OrderedDict

class PoseDataUtils:
    @staticmethod
    def create_empty_pose():
        return {
            "id": -1,
            "bbox": [0, 0, 0, 0],
            "confidence": 0,
            "keypoints": [[0, 0, 0] for _ in range(17)]
        }

    @staticmethod
    def validate_pose(pose):
        required_keys = ["id", "bbox", "confidence", "keypoints"]
        for key in required_keys:
            if key not in pose:
                raise ValueError(f"Missing required key: {key}")
        
        if len(pose["bbox"]) != 4:
            raise ValueError("bbox must have exactly 4 elements")
        
        if len(pose["keypoints"]) != 17:
            raise ValueError("keypoints must have exactly 17 elements")
        
        for keypoint in pose["keypoints"]:
            if len(keypoint) != 3:
                raise ValueError("Each keypoint must have exactly 3 elements")

    @staticmethod
    def format_poses(poses_dict, num_frames):
        formatted_poses = OrderedDict()
        
        for frame in range(num_frames):
            if frame in poses_dict and poses_dict[frame]:
                formatted_poses[frame] = poses_dict[frame]
                PoseDataUtils.validate_pose(formatted_poses[frame])
            else:
                formatted_poses[frame] = PoseDataUtils.create_empty_pose()
        
        return formatted_poses

    @staticmethod
    def save_poses(poses_dict, num_frames, output_file):
        if not poses_dict:
            print(f"Warning: Poses dictionary is empty. Not saving to {output_file}")
            return

        formatted_poses = PoseDataUtils.format_poses(poses_dict, num_frames)
        
        # Check if the formatted poses are not just empty frames
        if all(len(poses) == 1 and poses[0]['id'] == -1 for poses in formatted_poses.values()):
            print(f"Warning: All poses are empty. Not saving to {output_file}")
            return

        with open(output_file, 'w') as f:
            json.dump(formatted_poses, f, indent=2)

    @staticmethod
    def load_poses(json_path):
        with open(json_path, 'r') as f:
            detections = json.load(f)

        # Ensure all frame keys are integers
        return {int(frame): data for frame, data in detections.items()}
