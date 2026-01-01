"""
DeepLSD Line Detector Module

This module provides functionality to extract 2D line features from video frames
using the DeepLSD (Deep Line Segment Detector) method.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict


class DeepLSDLineDetector(nn.Module):
    """
    DeepLSD-based line detector for extracting 2D line features from images.
    
    This detector uses a convolutional neural network to detect line segments
    in images with sub-pixel accuracy.
    """
    
    def __init__(self, 
                 device: str = 'cuda',
                 min_length: float = 15.0,
                 max_num_lines: int = 500,
                 score_threshold: float = 0.5):
        """
        Initialize the DeepLSD line detector.
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
            min_length: Minimum length of detected lines in pixels
            max_num_lines: Maximum number of lines to return per image
            score_threshold: Minimum score threshold for line detection
        """
        super().__init__()
        self.device = device
        self.min_length = min_length
        self.max_num_lines = max_num_lines
        self.score_threshold = score_threshold
        
        # Initialize the line detection network
        self._build_network()
        
    def _build_network(self):
        """Build the DeepLSD neural network architecture."""
        # Simplified DeepLSD-inspired architecture
        # In practice, you would load a pre-trained DeepLSD model
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 3, 3, padding=1),  # Output: line heatmap, angle, and score
        )
        
        self.to(self.device)
        
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for line detection.
        
        Args:
            image: Input image as numpy array (H, W, 3) in BGR format
            
        Returns:
            Preprocessed image tensor (1, 3, H, W)
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            image_tensor: Input image tensor (B, 3, H, W)
            
        Returns:
            Network output with line predictions
        """
        features = self.encoder(image_tensor)
        output = self.decoder(features)
        return output
    
    def extract_lines_from_heatmap(self, 
                                   heatmap: torch.Tensor,
                                   image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Extract line segments from the predicted heatmap.
        
        Args:
            heatmap: Predicted line heatmap (1, 3, H, W)
            image_shape: Original image shape (H, W)
            
        Returns:
            Array of line segments (N, 4) where each row is [x1, y1, x2, y2]
        """
        # Extract line probability, angle, and score
        line_prob = torch.sigmoid(heatmap[0, 0]).cpu().numpy()
        line_angle = heatmap[0, 1].cpu().numpy()
        line_score = torch.sigmoid(heatmap[0, 2]).cpu().numpy()
        
        # Find line endpoints using non-maximum suppression
        # This is a simplified version - in practice, DeepLSD uses more sophisticated methods
        lines = []
        threshold = self.score_threshold
        
        # Get high-confidence pixels
        valid_mask = (line_prob > threshold) & (line_score > threshold)
        y_coords, x_coords = np.where(valid_mask)
        
        if len(x_coords) == 0:
            return np.array([]).reshape(0, 4)
        
        # Group nearby pixels into line segments using angle information
        lines = self._group_pixels_to_lines(
            x_coords, y_coords, 
            line_angle[y_coords, x_coords],
            line_score[y_coords, x_coords],
            image_shape
        )
        
        return lines
    
    def _group_pixels_to_lines(self,
                               x_coords: np.ndarray,
                               y_coords: np.ndarray,
                               angles: np.ndarray,
                               scores: np.ndarray,
                               image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Group detected pixels into line segments.
        
        Args:
            x_coords: X coordinates of detected pixels
            y_coords: Y coordinates of detected pixels
            angles: Line angles at each pixel
            scores: Confidence scores
            image_shape: Original image shape (H, W)
            
        Returns:
            Array of line segments (N, 4)
        """
        lines = []
        
        # Simple clustering approach
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]
        used = np.zeros(len(x_coords), dtype=bool)
        
        for idx in sorted_indices[:self.max_num_lines]:
            if used[idx]:
                continue
                
            x, y = x_coords[idx], y_coords[idx]
            angle = angles[idx]
            
            # Find nearby pixels with similar angle
            dx = x_coords - x
            dy = y_coords - y
            dist = np.sqrt(dx**2 + dy**2)
            angle_diff = np.abs(angles - angle)
            angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
            
            # Group pixels that are close and have similar orientation
            close_mask = (dist < 50) & (angle_diff < 0.3) & (~used)
            
            if np.sum(close_mask) < 2:
                continue
            
            # Fit a line through these pixels
            group_x = x_coords[close_mask]
            group_y = y_coords[close_mask]
            
            # Use PCA to find line direction
            if len(group_x) >= 2:
                mean_x, mean_y = np.mean(group_x), np.mean(group_y)
                centered = np.column_stack([group_x - mean_x, group_y - mean_y])
                
                # SVD to find principal direction
                _, _, vt = np.linalg.svd(centered, full_matrices=False)
                direction = vt[0]
                
                # Project points onto line direction
                projections = centered @ direction
                t_min, t_max = projections.min(), projections.max()
                
                # Calculate endpoints
                x1 = mean_x + t_min * direction[0]
                y1 = mean_y + t_min * direction[1]
                x2 = mean_x + t_max * direction[0]
                y2 = mean_y + t_max * direction[1]
                
                # Check line length
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                if length >= self.min_length:
                    # Clip to image boundaries
                    x1 = np.clip(x1, 0, image_shape[1] - 1)
                    y1 = np.clip(y1, 0, image_shape[0] - 1)
                    x2 = np.clip(x2, 0, image_shape[1] - 1)
                    y2 = np.clip(y2, 0, image_shape[0] - 1)
                    
                    lines.append([x1, y1, x2, y2])
                    used[close_mask] = True
        
        if len(lines) == 0:
            return np.array([]).reshape(0, 4)
        
        return np.array(lines)
    
    @torch.no_grad()
    def detect_lines(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect line segments in an image.
        
        Args:
            image: Input image as numpy array (H, W, 3)
            
        Returns:
            Tuple of (lines, scores) where:
                - lines: Array of line segments (N, 4) as [x1, y1, x2, y2]
                - scores: Confidence scores for each line (N,)
        """
        self.eval()
        
        # Preprocess image
        image_tensor = self.preprocess_image(image)
        image_shape = image.shape[:2]
        
        # Run network
        output = self.forward(image_tensor)
        
        # Extract lines
        lines = self.extract_lines_from_heatmap(output, image_shape)
        
        # Compute scores (simplified - use average along line)
        if len(lines) > 0:
            scores = np.ones(len(lines))  # Placeholder scores
        else:
            scores = np.array([])
        
        return lines, scores


class LSDLineDetector:
    """
    Line Segment Detector using OpenCV's LSD algorithm as a fallback.
    """
    
    def __init__(self, min_length: float = 15.0, max_num_lines: int = 500):
        """
        Initialize LSD detector.
        
        Args:
            min_length: Minimum length of detected lines
            max_num_lines: Maximum number of lines to return
        """
        self.min_length = min_length
        self.max_num_lines = max_num_lines
        self.lsd = cv2.createLineSegmentDetector(0)
    
    def detect_lines(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect lines using LSD algorithm.
        
        Args:
            image: Input image (H, W, 3) or (H, W)
            
        Returns:
            Tuple of (lines, scores)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect lines
        lines, width, prec, nfa = self.lsd.detect(gray)
        
        if lines is None:
            return np.array([]).reshape(0, 4), np.array([])
        
        # Reshape lines
        lines = lines.reshape(-1, 4)
        
        # Filter by length
        lengths = np.sqrt((lines[:, 2] - lines[:, 0])**2 + 
                         (lines[:, 3] - lines[:, 1])**2)
        valid_mask = lengths >= self.min_length
        lines = lines[valid_mask]
        
        # Sort by length and take top N
        if len(lines) > self.max_num_lines:
            lengths = lengths[valid_mask]
            top_indices = np.argsort(lengths)[::-1][:self.max_num_lines]
            lines = lines[top_indices]
        
        # Create scores based on length
        scores = np.sqrt((lines[:, 2] - lines[:, 0])**2 + 
                        (lines[:, 3] - lines[:, 1])**2) / 100.0
        
        return lines, scores
