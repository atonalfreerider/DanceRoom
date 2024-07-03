import os
from person_segmentation import PersonSegmentation
from room_tracker import detect_lines_and_axes


def main():
    input_video = '/home/john/Downloads/carlos-aline-spin-cam1.mp4'
    output_dir = '/home/john/Desktop/out'  # Replace with your desired output directory
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Segment people, create background-only video, and save masks and poses
    segmenter = PersonSegmentation(output_dir)
    if not segmenter.process_video(input_video):
        print("Error processing video. Exiting.")
        return

    bg_video_path = segmenter.get_bg_video_path()

    # Step 2: Process background-only video for room orientation and render poses
    final_output_path = os.path.join(output_dir, 'final_output.mp4')
    if os.path.exists(bg_video_path):
        detect_lines_and_axes(bg_video_path, final_output_path)
        print(f"Processing complete. Final output saved to: {final_output_path}")
    else:
        print(f"Error: Background video not found at {bg_video_path}")


if __name__ == "__main__":
    main()
