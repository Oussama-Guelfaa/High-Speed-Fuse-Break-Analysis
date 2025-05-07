"""
High-Speed Fuse Break Analysis

This script analyzes high-speed X-ray radiography videos of industrial fuses
to measure the distance between fuse elements during breaking events.

Usage:
    python main.py --video_path Camera_15_04_58.mp4 --output_dir results
"""

import argparse
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import pandas as pd
import time
import logging
from scipy.signal import savgol_filter

from video_processor import VideoProcessor
from image_processor import FuseImageProcessor
from visualization import FuseAnalysisVisualizer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='High-Speed Fuse Break Analysis')
    parser.add_argument('--video_path', type=str, default='Camera_15_04_58.mp4',
                        help='Path to the video file')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--calibration_frame', type=int, default=1,
                        help='Frame index to use for calibration')
    parser.add_argument('--calibration_value_mm', type=float, default=2.0,
                        help='Known height (H) of the fuse in mm')
    parser.add_argument('--save_frames', action='store_true',
                        help='Save processed frames')
    parser.add_argument('--create_video', action='store_true',
                        help='Create video with measurements')
    parser.add_argument('--fps', type=int, default=10,
                        help='Frames per second for output video')

    return parser.parse_args()


def main():
    """Main function to run the fuse break analysis."""
    # Parse arguments
    args = parse_arguments()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize components
    logger.info(f"Processing video: {args.video_path}")
    video_processor = VideoProcessor(args.video_path)
    image_processor = FuseImageProcessor(args.calibration_value_mm)
    visualizer = FuseAnalysisVisualizer(args.output_dir)

    # Load video and extract frames
    if not video_processor.load_video():
        logger.error("Failed to load video")
        return

    frames = video_processor.extract_all_frames()
    if not frames:
        logger.error("No frames extracted from video")
        return

    # Calibrate using the specified frame
    calibration_frame_idx = min(args.calibration_frame, len(frames) - 1)
    calibration_frame = frames[calibration_frame_idx].data

    try:
        pixels_per_mm = image_processor.calibrate(calibration_frame)
        logger.info(f"Calibration: {pixels_per_mm:.2f} pixels/mm")
    except Exception as e:
        logger.error(f"Calibration failed: {str(e)}")
        return

    # Process all frames to measure distances
    logger.info("Measuring distances in all frames...")
    frame_indices = []
    distances = []
    processed_frames = []

    for i, frame in enumerate(frames):
        try:
            distance = image_processor.measure_distance(frame.data)
            if distance is not None:
                frame_indices.append(i)
                distances.append(distance)

                # Create visualization if needed
                if args.save_frames or args.create_video:
                    vis_frame = image_processor.visualize_measurement(frame.data, distance)
                    processed_frames.append(vis_frame)

                    if args.save_frames and i % 10 == 0:  # Save every 10th frame
                        cv2.imwrite(
                            os.path.join(args.output_dir, f"frame_{i:04d}.png"),
                            vis_frame
                        )
        except Exception as e:
            logger.warning(f"Error processing frame {i}: {str(e)}")

    # Apply smoothing to the distance measurements
    if len(distances) > 5:
        # Use Savitzky-Golay filter for smoothing
        # The window size must be odd and less than the data length
        window_size = min(15, len(distances) - (len(distances) % 2 == 0))
        if window_size % 2 == 0:
            window_size -= 1
        if window_size >= 5:  # Minimum window size for the filter
            try:
                smoothed_distances = savgol_filter(distances, window_size, 3)
            except Exception as e:
                logger.warning(f"Smoothing failed: {str(e)}. Using original data.")
                smoothed_distances = distances
        else:
            smoothed_distances = distances
    else:
        smoothed_distances = distances

    # Save both original and smoothed distance measurements to CSV
    results_df = pd.DataFrame({
        'frame': frame_indices,
        'distance_mm': distances,
        'smoothed_distance_mm': smoothed_distances
    })
    csv_path = os.path.join(args.output_dir, 'distance_measurements.csv')
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Measurements saved to {csv_path}")

    # Create distance vs frame plot with both original and smoothed data
    plot_path = os.path.join(args.output_dir, 'distance_vs_frame.png')
    visualizer.plot_distance_vs_frame(
        frame_indices,
        smoothed_distances,  # Use smoothed data for the main plot
        original_distances=distances,  # Also show original data
        title="Fuse Breaking Distance vs Frame",
        save_path=plot_path
    )

    # Create comparison grid of key frames
    if processed_frames:
        # Select key frames (first, 25%, 50%, 75%, last)
        indices = [0, len(processed_frames)//4, len(processed_frames)//2,
                  3*len(processed_frames)//4, len(processed_frames)-1]
        key_frames = [processed_frames[i] for i in indices]
        key_titles = [f"Frame {frame_indices[i]}" for i in indices]

        grid_path = os.path.join(args.output_dir, 'key_frames.png')
        visualizer.create_comparison_grid(
            key_frames,
            key_titles,
            rows=1,
            cols=len(key_frames),
            save_path=grid_path
        )

    # Create video with measurements
    if args.create_video and processed_frames:
        video_path = os.path.join(args.output_dir, 'fuse_break_analysis.mp4')
        visualizer.create_video_with_measurements(
            processed_frames,
            distances,
            video_path,
            fps=args.fps
        )

    logger.info("Analysis complete!")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
