#!/usr/bin/env python3
"""
A dedicated module for performing motion detection on video frames.
"""

import cv2

class MotionDetector:
    """A class to handle motion detection using background subtraction."""
    
    def __init__(self):
        """Initializes the motion detector."""
        # Create the background subtractor object. This is the core of the motion detection.
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )

    def detect(self, frame):
        """
        Detects motion in a given frame.
        
        Args:
            frame: A single video frame (NumPy array).
            
        Returns:
            A tuple containing:
            - processed_frame (numpy.ndarray): The frame with motion rectangles drawn on it.
            - motion_detected (bool): True if motion was detected, False otherwise.
        """
        processed_frame = frame.copy()
        
        # 1. Apply the background subtractor to get a foreground mask.
        # This mask highlights areas that have changed from the background.
        fg_mask = self.background_subtractor.apply(processed_frame)
        
        # 2. Clean up the mask to reduce noise.
        # Thresholding makes the mask purely black and white.
        fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]
        # Dilating helps to fill in gaps in the detected areas.
        fg_mask = cv2.dilate(fg_mask, None, iterations=2)
        
        # 3. Find contours (outlines) of the white areas in the mask.
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_detected = False
        
        # 4. Loop over contours and draw rectangles if they are large enough.
        for contour in contours:
            # Ignore very small contours to avoid false positives from noise.
            if cv2.contourArea(contour) < 1000:
                continue
            
            motion_detected = True
            # Get the bounding box for the contour and draw it on the original frame.
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
        return processed_frame, motion_detected
