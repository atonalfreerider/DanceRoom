import os
import argparse
from person_segmentation import DanceSegmentation

def main(input_video, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Segment people, create background-only video, and save masks and poses
    segmentation = DanceSegmentation(input_video, output_dir)
    segmentation.process_video()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video for person segmentation and room orientation.")
    parser.add_argument("input_video", help="Path to the input video file")
    parser.add_argument("output_dir", help="Path to the output directory")
    args = parser.parse_args()

    main(args.input_video, args.output_dir)