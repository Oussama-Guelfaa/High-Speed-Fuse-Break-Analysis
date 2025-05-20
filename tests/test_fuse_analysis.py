"""
Tests for the fuse analysis module.
"""

import sys
import os
import unittest
import numpy as np

# Add the notebooks directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'notebooks'))

# Import the module to test
from fuse_analysis import VideoFrame, VideoProcessor, FuseImageProcessor


class TestFuseAnalysis(unittest.TestCase):
    """Test cases for the fuse analysis module."""

    def test_video_frame(self):
        """Test the VideoFrame class."""
        # Create a test frame
        frame_data = np.zeros((100, 100), dtype=np.uint8)
        frame = VideoFrame(index=1, data=frame_data, timestamp=0.1)
        
        # Check the frame properties
        self.assertEqual(frame.index, 1)
        self.assertEqual(frame.timestamp, 0.1)
        self.assertEqual(frame.data.shape, (100, 100))
    
    def test_fuse_image_processor_init(self):
        """Test the initialization of the FuseImageProcessor class."""
        processor = FuseImageProcessor(calibration_value_mm=2.0)
        
        # Check the processor properties
        self.assertEqual(processor.calibration_value_mm, 2.0)
        self.assertIsNone(processor.pixels_per_mm)
        self.assertFalse(processor.calibrated)
    
    def test_preprocess_image(self):
        """Test the image preprocessing function."""
        # Create a test image
        image = np.zeros((100, 100), dtype=np.uint8)
        image[40:60, 40:60] = 255  # Add a white square in the middle
        
        # Process the image
        processor = FuseImageProcessor()
        processed = processor.preprocess_image(image)
        
        # Check that the processed image is binary
        self.assertTrue(np.all(np.isin(processed, [0, 255])))
        
        # Check that the processed image has the same shape as the input
        self.assertEqual(processed.shape, image.shape)


if __name__ == '__main__':
    unittest.main()
