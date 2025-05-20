"""
High-Speed Fuse Break Analysis

This module contains the complete code for analyzing high-speed X-ray radiography videos
of industrial fuses to measure the distance between fuse elements during breaking events.
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import pandas as pd
import time
import logging
from dataclasses import dataclass
from scipy.signal import savgol_filter
from skimage import measure

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class VideoFrame:
    """Class to store video frame data and metadata."""
    index: int
    data: np.ndarray
    timestamp: Optional[float] = None


class VideoProcessor:
    """Class to process high-speed video of fuse breaking events."""

    def __init__(self, video_path: str):
        """
        Initialize the video processor.

        Args:
            video_path: Path to the video file
        """
        self.video_path = video_path
        self.video_obj = None
        self.frames = []
        self.height = 0
        self.width = 0
        self.fps = 0
        self.frame_count = 0

    def load_video(self) -> bool:
        """
        Load the video file and extract basic properties.

        Returns:
            bool: True if video loaded successfully, False otherwise
        """
        try:
            self.video_obj = cv2.VideoCapture(self.video_path)
            if not self.video_obj.isOpened():
                print(f"Error: Could not open video file {self.video_path}")
                return False

            # Get video properties
            self.width = int(self.video_obj.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.video_obj.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.video_obj.get(cv2.CAP_PROP_FPS)
            self.frame_count = int(self.video_obj.get(cv2.CAP_PROP_FRAME_COUNT))

            print(f"Video loaded: {self.width}x{self.height}, {self.fps} fps, {self.frame_count} frames")
            return True

        except Exception as e:
            print(f"Error loading video: {str(e)}")
            return False

    def extract_all_frames(self) -> List[VideoFrame]:
        """
        Extract all frames from the video and store them.

        Returns:
            List[VideoFrame]: List of video frames
        """
        if self.video_obj is None or not self.video_obj.isOpened():
            if not self.load_video():
                return []

        self.frames = []
        frame_idx = 0

        while True:
            ret, frame = self.video_obj.read()
            if not ret:
                break

            # Convert to grayscale if the frame is color
            if len(frame.shape) == 3:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray_frame = frame

            # Store frame
            self.frames.append(VideoFrame(
                index=frame_idx,
                data=gray_frame,
                timestamp=frame_idx / self.fps if self.fps > 0 else None
            ))

            frame_idx += 1

        print(f"Extracted {len(self.frames)} frames")
        return self.frames

    def get_frame(self, index: int) -> Optional[np.ndarray]:
        """
        Get a specific frame by index.

        Args:
            index: Frame index

        Returns:
            np.ndarray: Frame data or None if index is invalid
        """
        if 0 <= index < len(self.frames):
            return self.frames[index].data
        return None

    def release(self):
        """Release video resources."""
        if self.video_obj is not None and self.video_obj.isOpened():
            self.video_obj.release()


class FuseImageProcessor:
    """Class to process X-ray images of fuses and measure breaking distances."""

    def __init__(self, calibration_value_mm: float = 2.0):
        """
        Initialize the fuse image processor.

        Args:
            calibration_value_mm: The known height H of the fuse in mm (default: 2.0)
        """
        self.calibration_value_mm = calibration_value_mm
        self.pixels_per_mm = None
        self.calibrated = False

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image for better segmentation.

        Args:
            image: Input grayscale image

        Returns:
            np.ndarray: Preprocessed image
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)

        # Use Otsu's thresholding which is better for bimodal images like X-rays
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Apply morphological operations to clean up the binary image
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Apply closing to fill small holes
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

        return cleaned

    def segment_fuse_elements(self, image: np.ndarray) -> Tuple[np.ndarray, List]:
        """
        Segment the fuse elements (black parts) in the image.

        Args:
            image: Input grayscale image

        Returns:
            Tuple containing:
                - Binary mask of segmented elements
                - List of properties for each detected region
        """
        # Preprocess the image
        binary = self.preprocess_image(image)

        # Find connected components
        labeled_img = measure.label(binary)
        regions = measure.regionprops(labeled_img)

        # Filter regions by area to remove noise
        min_area = 100  # Adjust based on image resolution
        valid_regions = [r for r in regions if r.area >= min_area]

        # Create a mask with only the valid regions
        mask = np.zeros_like(binary)
        for region in valid_regions:
            for coord in region.coords:
                mask[coord[0], coord[1]] = 255

        return mask, valid_regions

    def calibrate(self, reference_image: np.ndarray) -> float:
        """
        Calibrate the measurement system using a reference image.

        Args:
            reference_image: Reference image where H is known

        Returns:
            float: Pixels per millimeter calibration factor
        """
        # Segment the fuse elements
        mask, regions = self.segment_fuse_elements(reference_image)

        if len(regions) < 1:
            raise ValueError("No fuse elements detected in reference image")

        # Find the height (H) in pixels
        # For calibration, we use the first frame where the fuse is intact
        heights = [r.bbox[2] - r.bbox[0] for r in regions]
        h_pixels = max(heights)  # Use the largest height as H

        # Calculate pixels per mm
        # Dividing by 10 times the calibration value effectively multiplies the distance scale by 10
        self.pixels_per_mm = h_pixels / (self.calibration_value_mm / 10)
        self.calibrated = True

        print(f"Calibration complete: {self.pixels_per_mm:.2f} pixels/mm (scale multiplied by 10)")
        return self.pixels_per_mm

    def measure_distance(self, image: np.ndarray) -> Optional[float]:
        """
        Measure the distance between fuse elements in millimeters.

        Args:
            image: Input grayscale image

        Returns:
            float: Distance in millimeters or None if measurement failed
        """
        if not self.calibrated:
            raise ValueError("Calibration required before measurement")

        # Segment the fuse elements
        mask, regions = self.segment_fuse_elements(image)

        # Check if we have enough regions to measure
        if len(regions) < 2:
            # If we have only one region, the fuse is likely intact
            if len(regions) == 1:
                # Check if the region spans most of the width
                region = regions[0]
                width = image.shape[1]
                region_width = region.bbox[3] - region.bbox[1]

                if region_width > 0.5 * width:
                    # This is likely an intact fuse
                    return 0.0

            # For frames before breaking starts
            # Check if this is an early frame (fuse intact)
            # Look for dark pixels in the middle of the image
            h, w = image.shape
            center_region = image[h//4:3*h//4, w//4:3*w//4]
            if np.mean(center_region) < 100:  # Adjust threshold as needed
                return 0.0

            return None

        # For frames with multiple regions, we need to identify the main fuse parts

        # Filter regions by size to focus on the main fuse parts
        min_area_ratio = 0.01  # Minimum area as a fraction of the largest region
        largest_area = max(r.area for r in regions)
        significant_regions = [r for r in regions if r.area > min_area_ratio * largest_area]

        if len(significant_regions) < 2:
            return 0.0  # Not enough significant regions

        # Sort regions horizontally (by x-coordinate)
        sorted_regions = sorted(significant_regions, key=lambda r: r.centroid[1])

        # Find the leftmost and rightmost significant regions
        left_regions = sorted_regions[:len(sorted_regions)//2]
        right_regions = sorted_regions[len(sorted_regions)//2:]

        if not left_regions or not right_regions:
            return 0.0

        # Find the rightmost point of all left regions
        left_edge = max(r.bbox[1] + r.bbox[3] for r in left_regions)  # rightmost edge of left regions

        # Find the leftmost point of all right regions
        right_edge = min(r.bbox[1] for r in right_regions)  # leftmost edge of right regions

        # Calculate distance
        distance_pixels = max(0, right_edge - left_edge)
        distance_mm = distance_pixels / self.pixels_per_mm

        # Note: The distance is already multiplied by 10 due to our calibration change

        # Apply a threshold to avoid noise
        if distance_mm < 0.1:  # Minimum meaningful distance
            return 0.0

        return distance_mm

    def visualize_measurement(self, image: np.ndarray, distance_mm: float) -> np.ndarray:
        """
        Create a visualization of the measurement on the image.

        Args:
            image: Input grayscale image
            distance_mm: Measured distance in millimeters

        Returns:
            np.ndarray: Visualization image with annotations
        """
        # Convert grayscale to color for visualization
        vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Segment the fuse elements
        mask, regions = self.segment_fuse_elements(image)

        # Draw contours around detected regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_img, contours, -1, (0, 255, 0), 2)

        # Add distance measurement text
        cv2.putText(
            vis_img,
            f"d = {distance_mm:.3f} mm (échelle x10)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        return vis_img


class FuseAnalysisVisualizer:
    """Class to create visualizations of fuse breaking analysis."""

    def __init__(self, output_dir: str):
        """
        Initialize the visualizer.

        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_distance_vs_frame(self,
                              frame_indices: List[int],
                              distances: List[float],
                              original_distances: Optional[List[float]] = None,
                              title: str = "Distance vs Frame",
                              save_path: Optional[str] = None) -> None:
        """
        Plot the distance between fuse elements vs frame number.

        Args:
            frame_indices: List of frame indices
            distances: List of corresponding distances in mm
            original_distances: Optional list of original (unsmoothed) distances
            title: Plot title
            save_path: Path to save the plot (if None, plot is displayed)
        """
        plt.figure(figsize=(10, 6))

        # Plot the main distance data
        plt.plot(frame_indices, distances, 'b-', linewidth=2, label='Distance')

        # If original data is provided, plot it as a lighter line
        if original_distances is not None:
            plt.plot(frame_indices, original_distances, 'r-', linewidth=1, alpha=0.5, label='Original Data')
            plt.legend()

        plt.xlabel('Frame')
        plt.ylabel('Distance (mm) - Échelle x10')
        plt.title(f"{title} - Échelle multipliée par 10")
        plt.grid(True)

        # Set y-axis to start at 0
        plt.ylim(bottom=0)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def create_comparison_grid(self,
                              images: List[np.ndarray],
                              titles: List[str],
                              rows: int = 1,
                              cols: Optional[int] = None,
                              figsize: Tuple[int, int] = (15, 5),
                              save_path: Optional[str] = None) -> None:
        """
        Create a grid of images for comparison.

        Args:
            images: List of images to display
            titles: List of titles for each image
            rows: Number of rows in the grid
            cols: Number of columns in the grid (calculated if None)
            figsize: Figure size (width, height) in inches
            save_path: Path to save the figure (if None, figure is displayed)
        """
        n_images = len(images)
        if cols is None:
            cols = (n_images + rows - 1) // rows

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows * cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, (img, title) in enumerate(zip(images, titles)):
            if i < len(axes):
                if len(img.shape) == 2 or img.shape[2] == 1:
                    # Grayscale image
                    axes[i].imshow(img, cmap='gray')
                else:
                    # Color image (BGR to RGB for matplotlib)
                    axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                axes[i].set_title(title)
                axes[i].axis('off')

        # Hide unused subplots
        for i in range(n_images, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison grid saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def create_video_with_measurements(self,
                                      frames: List[np.ndarray],
                                      distances: List[float],
                                      output_path: str,
                                      fps: int = 10) -> None:
        """
        Create a video with distance measurements overlaid.

        Args:
            frames: List of frames with visualizations
            distances: List of corresponding distances
            output_path: Path to save the output video
            fps: Frames per second for the output video
        """
        if not frames:
            print("No frames to create video")
            return

        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for i, (frame, distance) in enumerate(zip(frames, distances)):
            # Convert grayscale to color if needed
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # Add frame number
            cv2.putText(
                frame,
                f"Frame: {i}",
                (width - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            video_writer.write(frame)

        video_writer.release()
        print(f"Video saved to {output_path}")


def main():
    """Main function to run the fuse break analysis."""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze high-speed fuse break videos')
    parser.add_argument('--video_path', type=str, default='data/Camera_15_04_58.mp4',
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

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize components
    video_processor = VideoProcessor(args.video_path)
    image_processor = FuseImageProcessor(args.calibration_value_mm)
    visualizer = FuseAnalysisVisualizer(args.output_dir)

    # Process video
    logger.info(f"Processing video: {args.video_path}")
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
        title="Distance entre les éléments du fusible en fonction du temps",
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
