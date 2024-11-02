import os
import argparse

import yolo_pose
from segmenter import Segmenter
from manual_role_assignment import ManualRoleAssignment
from manual_review import ManualReview
from temporal_smoothing import TemporalSmoothing
#import room_tracker
from dancer_tracker import DancerTracker
from debug_video import DebugVideo

def main(input_video, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # use depth maps from UniDepth to isolate figures dancing in center of floor
    #segmenter = Segmenter(input_video, output_dir)
    #segmenter.process_video()

    # use YOLOv11 to detect poses of figures and track them
    yoloPose = yolo_pose.YOLOPose(output_dir + "/figure-masks", output_dir + "/detections.json")
    yoloPose.detect_poses()

    dancer_tracker = DancerTracker(input_video, output_dir)
    dancer_tracker.process_video()
    
    # Create debug video (optional)
    #dancer_tracker.create_debug_video()

    # manually assign roles to tracked persons
    #manual_assigner = ManualRoleAssignment(input_video, output_dir + "/detections.json", output_dir)
    #manual_assigner.process_tracks()

    #room_tracker.room_tracker(input_video, output_dir)
    #room_tracker.debug_video(input_video, output_dir, output_dir + "/deltas.json")

    # detect dancer gender and assign to lead or follow


    #manual_review = ManualReview(input_video, output_dir + "/detections.json", output_dir)
    #manual_review.run()

    # Apply temporal smoothing to lead and follow keypoints
    #smoother = TemporalSmoothing(output_dir)
    #smoother.run()

    debug_video = DebugVideo(input_video, output_dir)
    debug_video.generate_debug_video()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video for person segmentation and room orientation.")
    parser.add_argument("input_video", help="Path to the input video file")
    parser.add_argument("output_dir", help="Path to the output directory")
    args = parser.parse_args()

    main(args.input_video, args.output_dir)
