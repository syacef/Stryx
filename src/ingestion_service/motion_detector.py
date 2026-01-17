"""
Motion detection using frame differencing.
"""
import cv2
import numpy as np
from typing import Tuple
from config import MOTION_THRESHOLD, MIN_MOTION_AREA


class MotionDetector:
    """
    Simple motion detection using frame differencing.
    Uses background subtraction to detect movement between frames.
    """
    
    def __init__(self, threshold: float = MOTION_THRESHOLD, min_area: int = MIN_MOTION_AREA):
        """
        Initialize motion detector.
        
        Args:
            threshold: Pixel difference threshold for motion detection
            min_area: Minimum contour area to consider as motion
        """
        self.threshold = threshold
        self.min_area = min_area
        self.prev_frame = None
    
    def detect_motion(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Detect motion between current frame and previous frame.
        
        Args:
            frame: Current frame (BGR format)
        
        Returns:
            Tuple of (has_motion, motion_score)
                has_motion: Boolean indicating if motion was detected
                motion_score: Numerical score representing amount of motion
        """
        # Convert to grayscale and blur to reduce noise
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Initialize first frame
        if self.prev_frame is None:
            self.prev_frame = gray
            return False, 0.0
        
        # Compute absolute difference between frames
        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        
        # Dilate to fill gaps in detected motion
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours of moving objects
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate total motion score (sum of significant contour areas)
        motion_score = sum(
            cv2.contourArea(c) 
            for c in contours 
            if cv2.contourArea(c) > self.min_area
        )
        has_motion = motion_score > 0
        
        # Update previous frame for next comparison
        self.prev_frame = gray
        
        return has_motion, motion_score
    
    def reset(self):
        """Reset detector state (clears previous frame)."""
        self.prev_frame = None
