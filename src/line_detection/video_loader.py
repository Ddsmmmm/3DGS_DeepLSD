"""
Video frame loader and preprocessor for line detection.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Generator
from pathlib import Path


class VideoFrameLoader:
    """
    Utility class for loading and preprocessing video frames for line detection.
    """
    
    def __init__(self, 
                 video_path: Optional[str] = None,
                 image_dir: Optional[str] = None,
                 target_size: Optional[Tuple[int, int]] = None,
                 frame_skip: int = 1):
        """
        Initialize video frame loader.
        
        Args:
            video_path: Path to video file
            image_dir: Path to directory containing images
            target_size: Target size for resizing frames (width, height)
            frame_skip: Number of frames to skip between processed frames
        """
        self.video_path = video_path
        self.image_dir = image_dir
        self.target_size = target_size
        self.frame_skip = frame_skip
        
        if video_path is None and image_dir is None:
            raise ValueError("Either video_path or image_dir must be provided")
        
        if video_path is not None and image_dir is not None:
            raise ValueError("Only one of video_path or image_dir should be provided")
        
        # Initialize frame count
        self._frame_count = None
        
    def get_frame_count(self) -> int:
        """Get total number of frames."""
        if self._frame_count is not None:
            return self._frame_count
        
        if self.video_path is not None:
            cap = cv2.VideoCapture(self.video_path)
            self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        else:
            image_files = self._get_image_files()
            self._frame_count = len(image_files)
        
        return self._frame_count
    
    def _get_image_files(self) -> List[str]:
        """Get sorted list of image files."""
        if self.image_dir is None:
            return []
        
        image_dir = Path(self.image_dir)
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        image_files = []
        for ext in extensions:
            image_files.extend(image_dir.glob(f'*{ext}'))
            image_files.extend(image_dir.glob(f'*{ext.upper()}'))
        
        return sorted([str(f) for f in image_files])
    
    def load_frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Load frames from video or images.
        
        Yields:
            Tuple of (frame_index, frame_array)
        """
        if self.video_path is not None:
            yield from self._load_from_video()
        else:
            yield from self._load_from_images()
    
    def _load_from_video(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Load frames from video file."""
        cap = cv2.VideoCapture(self.video_path)
        
        frame_idx = 0
        processed_idx = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if frame_idx % self.frame_skip == 0:
                if self.target_size is not None:
                    frame = cv2.resize(frame, self.target_size)
                
                yield processed_idx, frame
                processed_idx += 1
            
            frame_idx += 1
        
        cap.release()
    
    def _load_from_images(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Load frames from image directory."""
        image_files = self._get_image_files()
        
        for idx, image_path in enumerate(image_files):
            if idx % self.frame_skip != 0:
                continue
            
            frame = cv2.imread(image_path)
            
            if frame is None:
                print(f"Warning: Could not load image {image_path}")
                continue
            
            if self.target_size is not None:
                frame = cv2.resize(frame, self.target_size)
            
            yield idx // self.frame_skip, frame
    
    def load_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Load a specific frame by index.
        
        Args:
            frame_idx: Frame index to load
            
        Returns:
            Frame array or None if frame not found
        """
        if self.video_path is not None:
            cap = cv2.VideoCapture(self.video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx * self.frame_skip)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return None
            
            if self.target_size is not None:
                frame = cv2.resize(frame, self.target_size)
            
            return frame
        else:
            image_files = self._get_image_files()
            actual_idx = frame_idx * self.frame_skip
            
            if actual_idx >= len(image_files):
                return None
            
            frame = cv2.imread(image_files[actual_idx])
            
            if frame is not None and self.target_size is not None:
                frame = cv2.resize(frame, self.target_size)
            
            return frame


def preprocess_frame_for_detection(frame: np.ndarray,
                                   denoise: bool = True,
                                   enhance_contrast: bool = True) -> np.ndarray:
    """
    Preprocess frame for better line detection.
    
    Args:
        frame: Input frame (H, W, 3)
        denoise: Whether to apply denoising
        enhance_contrast: Whether to enhance contrast
        
    Returns:
        Preprocessed frame
    """
    processed = frame.copy()
    
    if denoise:
        processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)
    
    if enhance_contrast:
        # Convert to LAB color space
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge and convert back to BGR
        lab = cv2.merge([l, a, b])
        processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return processed
