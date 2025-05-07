"""
Image Processing Module for High-Speed Fuse Break Analysis.

This module provides functionality to analyze X-ray radiography images
of fuse breaking events, measuring the distance between fuse elements.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt
from skimage import filters, measure, morphology


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
        
        # Apply adaptive thresholding to handle uneven illumination
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Apply morphological operations to clean up the binary image
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return cleaned
    
    def segment_fuse_elements(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
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
        self.pixels_per_mm = h_pixels / self.calibration_value_mm
        self.calibrated = True
        
        print(f"Calibration complete: {self.pixels_per_mm:.2f} pixels/mm")
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
        
        if len(regions) < 2:
            # If less than 2 regions, the fuse might be intact or completely broken
            if len(regions) == 1:
                # Fuse is intact, distance is 0
                return 0.0
            else:
                # No fuse elements detected
                return None
        
        # Sort regions by x-coordinate
        sorted_regions = sorted(regions, key=lambda r: r.centroid[1])
        
        # Calculate distance between the two main regions
        # We assume the two largest regions are the fuse elements
        if len(sorted_regions) >= 2:
            # Get the two largest regions
            largest_regions = sorted(sorted_regions, key=lambda r: r.area, reverse=True)[:2]
            # Sort them by x-coordinate
            left_right = sorted(largest_regions, key=lambda r: r.centroid[1])
            
            # Calculate the distance between the right edge of left region and left edge of right region
            left_region, right_region = left_right
            left_edge = left_region.bbox[1] + left_region.bbox[3]  # right edge of left region
            right_edge = right_region.bbox[1]  # left edge of right region
            
            distance_pixels = max(0, right_edge - left_edge)
            distance_mm = distance_pixels / self.pixels_per_mm
            
            return distance_mm
        
        return None
    
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
            f"d = {distance_mm:.3f} mm", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 0, 255), 
            2
        )
        
        return vis_img
