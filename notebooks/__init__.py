"""
High-Speed Fuse Break Analysis package.

This package contains modules for analyzing high-speed X-ray radiography videos
of industrial fuses to measure the distance between fuse elements during breaking events.
"""

# Import main classes for easier access
try:
    from .fuse_analysis import VideoFrame, VideoProcessor, FuseImageProcessor, FuseAnalysisVisualizer
except ImportError:
    # This allows the module to be imported even if some dependencies are missing
    pass
