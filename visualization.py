"""
Visualization Module for High-Speed Fuse Break Analysis.

This module provides functionality to visualize the results of fuse break analysis,
including distance measurements and key frames.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import cv2
import os


class FuseAnalysisVisualizer:
    """Class to visualize fuse break analysis results."""

    def __init__(self, output_dir: str = "results"):
        """
        Initialize the visualizer.

        Args:
            output_dir: Directory to save visualization outputs
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
        plt.ylabel('Distance (mm)')
        plt.title(title)
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
        Create a video with distance measurements overlaid on frames.

        Args:
            frames: List of frames
            distances: List of corresponding distances in mm
            output_path: Path to save the output video
            fps: Frames per second for the output video
        """
        if not frames:
            print("No frames provided for video creation")
            return

        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for i, (frame, distance) in enumerate(zip(frames, distances)):
            # Convert grayscale to color if needed
            if len(frame.shape) == 2 or frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # Add frame number and distance
            cv2.putText(
                frame,
                f"Frame: {i}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Distance: {distance:.3f} mm",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            video_writer.write(frame)

        video_writer.release()
        print(f"Video saved to {output_path}")
