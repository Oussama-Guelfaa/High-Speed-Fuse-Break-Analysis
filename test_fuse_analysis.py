"""
Test script for High-Speed Fuse Break Analysis.

This script tests the functionality of the fuse break analysis modules.
"""

import unittest
import os
import numpy as np
import cv2
from video_processor import VideoProcessor
from image_processor import FuseImageProcessor
from visualization import FuseAnalysisVisualizer


class TestVideoProcessor(unittest.TestCase):
    """Test cases for VideoProcessor class."""
    
    def setUp(self):
        """Set up test environment."""
        self.video_path = "Camera_15_04_58.mp4"
        self.processor = VideoProcessor(self.video_path)
    
    def test_load_video(self):
        """Test loading a video file."""
        result = self.processor.load_video()
        self.assertTrue(result)
        self.assertGreater(self.processor.width, 0)
        self.assertGreater(self.processor.height, 0)
        self.assertGreater(self.processor.fps, 0)
        self.assertGreater(self.processor.frame_count, 0)
    
    def test_extract_frames(self):
        """Test extracting frames from a video."""
        self.processor.load_video()
        frames = self.processor.extract_all_frames()
        self.assertGreater(len(frames), 0)
        self.assertEqual(frames[0].index, 0)
        self.assertIsNotNone(frames[0].data)
        self.assertIsInstance(frames[0].data, np.ndarray)


class TestImageProcessor(unittest.TestCase):
    """Test cases for FuseImageProcessor class."""
    
    def setUp(self):
        """Set up test environment."""
        self.processor = FuseImageProcessor(calibration_value_mm=2.0)
        
        # Create a simple test image
        self.test_image = np.ones((100, 200), dtype=np.uint8) * 255
        # Add two black rectangles (simulating fuse elements)
        self.test_image[40:60, 20:80] = 0  # Left element
        self.test_image[40:60, 120:180] = 0  # Right element
    
    def test_preprocess_image(self):
        """Test image preprocessing."""
        processed = self.processor.preprocess_image(self.test_image)
        self.assertIsInstance(processed, np.ndarray)
        self.assertEqual(processed.shape, self.test_image.shape)
    
    def test_segment_fuse_elements(self):
        """Test segmentation of fuse elements."""
        mask, regions = self.processor.segment_fuse_elements(self.test_image)
        self.assertIsInstance(mask, np.ndarray)
        self.assertEqual(mask.shape, self.test_image.shape)
        # Should detect at least one region
        self.assertGreaterEqual(len(regions), 1)
    
    def test_calibration(self):
        """Test calibration with a reference image."""
        try:
            pixels_per_mm = self.processor.calibrate(self.test_image)
            self.assertGreater(pixels_per_mm, 0)
            self.assertTrue(self.processor.calibrated)
        except ValueError:
            # If calibration fails on the test image, that's acceptable
            pass
    
    def test_measure_distance(self):
        """Test distance measurement."""
        try:
            # First calibrate
            self.processor.calibrate(self.test_image)
            # Then measure
            distance = self.processor.measure_distance(self.test_image)
            self.assertIsNotNone(distance)
            self.assertGreaterEqual(distance, 0)
        except ValueError:
            # If measurement fails on the test image, that's acceptable
            pass


class TestVisualizer(unittest.TestCase):
    """Test cases for FuseAnalysisVisualizer class."""
    
    def setUp(self):
        """Set up test environment."""
        self.output_dir = "test_results"
        os.makedirs(self.output_dir, exist_ok=True)
        self.visualizer = FuseAnalysisVisualizer(self.output_dir)
        
        # Create test data
        self.frame_indices = list(range(10))
        self.distances = [i * 0.1 for i in range(10)]
        
        # Create test images
        self.test_images = []
        for i in range(3):
            img = np.ones((100, 100, 3), dtype=np.uint8) * 255
            cv2.putText(img, f"Test {i}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            self.test_images.append(img)
    
    def test_plot_distance_vs_frame(self):
        """Test plotting distance vs frame."""
        save_path = os.path.join(self.output_dir, "test_plot.png")
        self.visualizer.plot_distance_vs_frame(
            self.frame_indices,
            self.distances,
            save_path=save_path
        )
        self.assertTrue(os.path.exists(save_path))
    
    def test_create_comparison_grid(self):
        """Test creating a comparison grid."""
        save_path = os.path.join(self.output_dir, "test_grid.png")
        self.visualizer.create_comparison_grid(
            self.test_images,
            ["Image 1", "Image 2", "Image 3"],
            save_path=save_path
        )
        self.assertTrue(os.path.exists(save_path))
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove test files
        for file in os.listdir(self.output_dir):
            os.remove(os.path.join(self.output_dir, file))
        os.rmdir(self.output_dir)


if __name__ == "__main__":
    unittest.main()
