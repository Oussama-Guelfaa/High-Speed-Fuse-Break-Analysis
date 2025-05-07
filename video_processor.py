"""
Video Processing Module for High-Speed Fuse Break Analysis.

This module provides functionality to read and process high-speed video frames
of fuse breaking events captured with X-ray radiography.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


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
