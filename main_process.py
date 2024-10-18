import os
import argparse

import yolo_pose
from segmenter import Segmenter
#import room_tracker
#from dancer_tracker import DancerTracker
from debug_video import DebugVideo

def main(input_video, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # use depth maps from UniDepth to isolate figures dancing in center of floor
    segmenter = Segmenter(input_video, output_dir)
    segmenter.process_video()

    # use YOLOv11 to detect poses of figures and track them
    yoloPose = yolo_pose.YOLOPose(output_dir + "/figure-masks", output_dir + "/detections.json")
    yoloPose.detect_poses()

    #room_tracker.room_tracker(input_video, output_dir)
    #room_tracker.debug_video(input_video, output_dir, output_dir + "/deltas.json")

    # detect dancer gender and assign to lead or follow
    #dancer_tracker = DancerTracker(input_video, output_dir)
    #dancer_tracker.process_video()

    

    debug_video = DebugVideo(input_video, output_dir)
    debug_video.generate_debug_video()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video for person segmentation and room orientation.")
    parser.add_argument("input_video", help="Path to the input video file")
    parser.add_argument("output_dir", help="Path to the output directory")
    args = parser.parse_args()

    main(args.input_video, args.output_dir)