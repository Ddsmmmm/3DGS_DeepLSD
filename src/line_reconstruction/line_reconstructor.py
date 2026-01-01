"""
3D Line Reconstruction Module

This module reconstructs 3D lines from 2D line detections across multiple views
by triangulating matched lines and aligning them with the 3DGS coordinate system.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.spatial.transform import Rotation
import cv2


class CameraPose:
    """Represents a camera pose with intrinsics and extrinsics."""
    
    def __init__(self,
                 K: np.ndarray,
                 R: np.ndarray,
                 t: np.ndarray,
                 frame_id: int):
        """
        Initialize camera pose.
        
        Args:
            K: Camera intrinsic matrix (3, 3)
            R: Rotation matrix (3, 3)
            t: Translation vector (3,) or (3, 1)
            frame_id: Frame identifier
        """
        self.K = K
        self.R = R
        self.t = t.reshape(3, 1) if t.shape == (3,) else t
        self.frame_id = frame_id
        
        # Compute projection matrix
        self.P = K @ np.hstack([R, t.reshape(3, 1)])
        
        # Compute camera center in world coordinates
        self.C = -R.T @ t.reshape(3, 1)
    
    def project_point(self, X: np.ndarray) -> np.ndarray:
        """
        Project 3D point to image plane.
        
        Args:
            X: 3D point (3,) or (3, N)
            
        Returns:
            2D projection (2,) or (2, N)
        """
        if X.ndim == 1:
            X = X.reshape(3, 1)
        
        x_h = self.P @ np.vstack([X, np.ones((1, X.shape[1]))])
        x = x_h[:2] / x_h[2]
        
        return x.squeeze()
    
    def backproject_ray(self, x: np.ndarray) -> np.ndarray:
        """
        Backproject 2D point to a 3D ray.
        
        Args:
            x: 2D point (2,)
            
        Returns:
            Ray direction in world coordinates (3,)
        """
        # Convert to normalized image coordinates
        x_h = np.array([x[0], x[1], 1.0])
        x_norm = np.linalg.inv(self.K) @ x_h
        
        # Transform to world coordinates
        ray_dir = self.R.T @ x_norm
        ray_dir = ray_dir / np.linalg.norm(ray_dir)
        
        return ray_dir


class Line2D:
    """Represents a 2D line segment in an image."""
    
    def __init__(self, 
                 endpoints: np.ndarray,
                 frame_id: int,
                 score: float = 1.0):
        """
        Initialize 2D line.
        
        Args:
            endpoints: Line endpoints (4,) as [x1, y1, x2, y2]
            frame_id: Frame identifier
            score: Detection confidence score
        """
        self.endpoints = endpoints
        self.frame_id = frame_id
        self.score = score
        
        # Compute line properties
        self.start = endpoints[:2]
        self.end = endpoints[2:]
        self.midpoint = (self.start + self.end) / 2
        self.direction = self.end - self.start
        self.length = np.linalg.norm(self.direction)
        self.direction = self.direction / (self.length + 1e-8)


class Line3D:
    """Represents a 3D line segment in world coordinates."""
    
    def __init__(self,
                 start: np.ndarray,
                 end: np.ndarray,
                 score: float = 1.0,
                 support_views: Optional[List[int]] = None):
        """
        Initialize 3D line.
        
        Args:
            start: Start point in 3D (3,)
            end: End point in 3D (3,)
            score: Reconstruction confidence score
            support_views: List of frame IDs that support this line
        """
        self.start = start
        self.end = end
        self.score = score
        self.support_views = support_views or []
        
        # Compute line properties
        self.midpoint = (start + end) / 2
        self.direction = end - start
        self.length = np.linalg.norm(self.direction)
        self.direction = self.direction / (self.length + 1e-8)
    
    def to_plucker(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert to Plücker coordinates representation.
        
        Returns:
            Tuple of (direction, moment) vectors
        """
        direction = self.direction
        moment = np.cross(self.start, self.end)
        return direction, moment


class LineReconstructor:
    """Reconstructs 3D lines from 2D line detections across multiple views."""
    
    def __init__(self,
                 reprojection_threshold: float = 5.0,
                 min_triangulation_angle: float = 5.0,
                 min_support_views: int = 2):
        """
        Initialize line reconstructor.
        
        Args:
            reprojection_threshold: Maximum reprojection error in pixels
            min_triangulation_angle: Minimum angle between views for triangulation (degrees)
            min_support_views: Minimum number of views supporting a 3D line
        """
        self.reprojection_threshold = reprojection_threshold
        self.min_triangulation_angle = np.deg2rad(min_triangulation_angle)
        self.min_support_views = min_support_views
    
    def triangulate_line(self,
                        line1: Line2D,
                        line2: Line2D,
                        pose1: CameraPose,
                        pose2: CameraPose) -> Optional[Line3D]:
        """
        Triangulate a 3D line from two 2D line observations.
        
        Args:
            line1: 2D line in first view
            line2: 2D line in second view
            pose1: Camera pose for first view
            pose2: Camera pose for second view
            
        Returns:
            Reconstructed 3D line or None if triangulation fails
        """
        # Check triangulation angle
        baseline = pose2.C - pose1.C
        baseline = baseline / np.linalg.norm(baseline)
        
        ray1_start = pose1.backproject_ray(line1.start)
        ray1_end = pose1.backproject_ray(line1.end)
        
        angle1 = np.arccos(np.clip(np.abs(baseline.T @ ray1_start), -1, 1))
        angle2 = np.arccos(np.clip(np.abs(baseline.T @ ray1_end), -1, 1))
        
        if min(angle1, angle2) < self.min_triangulation_angle:
            return None
        
        # Triangulate start and end points
        start_3d = self._triangulate_point(
            line1.start, line2.start, pose1, pose2
        )
        end_3d = self._triangulate_point(
            line1.end, line2.end, pose1, pose2
        )
        
        if start_3d is None or end_3d is None:
            return None
        
        # Compute average score
        score = (line1.score + line2.score) / 2
        
        # Create 3D line
        line_3d = Line3D(
            start_3d,
            end_3d,
            score,
            support_views=[line1.frame_id, line2.frame_id]
        )
        
        return line_3d
    
    def _triangulate_point(self,
                          pt1: np.ndarray,
                          pt2: np.ndarray,
                          pose1: CameraPose,
                          pose2: CameraPose) -> Optional[np.ndarray]:
        """
        Triangulate a 3D point from two 2D observations using DLT.
        
        Args:
            pt1: 2D point in first view (2,)
            pt2: 2D point in second view (2,)
            pose1: Camera pose for first view
            pose2: Camera pose for second view
            
        Returns:
            3D point (3,) or None if triangulation fails
        """
        # Build DLT matrix
        A = np.zeros((4, 4))
        A[0] = pt1[0] * pose1.P[2] - pose1.P[0]
        A[1] = pt1[1] * pose1.P[2] - pose1.P[1]
        A[2] = pt2[0] * pose2.P[2] - pose2.P[0]
        A[3] = pt2[1] * pose2.P[2] - pose2.P[1]
        
        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        X_h = Vt[-1]
        
        # Check if point is at infinity
        if np.abs(X_h[3]) < 1e-8:
            return None
        
        X = X_h[:3] / X_h[3]
        
        # Check reprojection error
        x1_reproj = pose1.project_point(X)
        x2_reproj = pose2.project_point(X)
        
        error1 = np.linalg.norm(x1_reproj - pt1)
        error2 = np.linalg.norm(x2_reproj - pt2)
        
        if error1 > self.reprojection_threshold or error2 > self.reprojection_threshold:
            return None
        
        return X
    
    def match_lines_across_views(self,
                                 lines1: List[Line2D],
                                 lines2: List[Line2D],
                                 pose1: CameraPose,
                                 pose2: CameraPose,
                                 descriptor_threshold: float = 0.8) -> List[Tuple[int, int]]:
        """
        Match lines between two views based on geometric and appearance similarity.
        
        Args:
            lines1: Lines from first view
            lines2: Lines from second view
            pose1: Camera pose for first view
            pose2: Camera pose for second view
            descriptor_threshold: Threshold for descriptor matching
            
        Returns:
            List of matched line pairs as (index1, index2)
        """
        if len(lines1) == 0 or len(lines2) == 0:
            return []
        
        matches = []
        
        # Compute epipolar geometry
        F = self._compute_fundamental_matrix(pose1, pose2)
        
        # For each line in view 1, find best match in view 2
        for i, line1 in enumerate(lines1):
            best_score = -1
            best_j = -1
            
            for j, line2 in enumerate(lines2):
                # Compute geometric compatibility using epipolar constraint
                epipolar_score = self._epipolar_line_score(line1, line2, F)
                
                if epipolar_score < 0.5:  # Low geometric compatibility
                    continue
                
                # Compute orientation similarity
                # (In practice, you would also use line descriptors)
                orientation_score = self._orientation_similarity(line1, line2)
                
                # Combined score
                score = 0.6 * epipolar_score + 0.4 * orientation_score
                
                if score > best_score and score > descriptor_threshold:
                    best_score = score
                    best_j = j
            
            if best_j >= 0:
                matches.append((i, best_j))
        
        return matches
    
    def _compute_fundamental_matrix(self,
                                   pose1: CameraPose,
                                   pose2: CameraPose) -> np.ndarray:
        """Compute fundamental matrix between two views."""
        # E = [t]_x * R
        t = pose2.R @ pose1.C + pose2.t - pose2.C
        t_x = np.array([
            [0, -t[2, 0], t[1, 0]],
            [t[2, 0], 0, -t[0, 0]],
            [-t[1, 0], t[0, 0], 0]
        ])
        
        R_rel = pose2.R @ pose1.R.T
        E = t_x @ R_rel
        
        # F = K2^-T * E * K1^-1
        F = np.linalg.inv(pose2.K).T @ E @ np.linalg.inv(pose1.K)
        
        return F
    
    def _epipolar_line_score(self,
                            line1: Line2D,
                            line2: Line2D,
                            F: np.ndarray) -> float:
        """
        Compute epipolar compatibility score between two lines.
        
        Args:
            line1: Line in first view
            line2: Line in second view
            F: Fundamental matrix
            
        Returns:
            Compatibility score in [0, 1]
        """
        # Compute epipolar lines
        l_start = F @ np.append(line1.start, 1)
        l_end = F @ np.append(line1.end, 1)
        
        # Compute distance from line2 endpoints to epipolar lines
        dist_start = np.abs(l_start @ np.append(line2.start, 1)) / np.linalg.norm(l_start[:2])
        dist_end = np.abs(l_end @ np.append(line2.end, 1)) / np.linalg.norm(l_end[:2])
        
        # Convert distance to score
        max_dist = 10.0  # pixels
        score = np.exp(-0.5 * ((dist_start + dist_end) / max_dist) ** 2)
        
        return score
    
    def _orientation_similarity(self, line1: Line2D, line2: Line2D) -> float:
        """
        Compute orientation similarity between two lines.
        
        Returns:
            Similarity score in [0, 1]
        """
        # Compute angle between line directions
        cos_angle = np.abs(np.dot(line1.direction, line2.direction))
        return cos_angle
    
    def reconstruct_lines(self,
                         lines_2d: Dict[int, List[Line2D]],
                         camera_poses: Dict[int, CameraPose]) -> List[Line3D]:
        """
        Reconstruct 3D lines from 2D detections across multiple views.
        
        Args:
            lines_2d: Dictionary mapping frame_id to list of 2D lines
            camera_poses: Dictionary mapping frame_id to camera poses
            
        Returns:
            List of reconstructed 3D lines
        """
        lines_3d = []
        
        # Get sorted frame IDs
        frame_ids = sorted(lines_2d.keys())
        
        # Match lines across consecutive views
        for i in range(len(frame_ids) - 1):
            frame_id1 = frame_ids[i]
            frame_id2 = frame_ids[i + 1]
            
            if frame_id1 not in camera_poses or frame_id2 not in camera_poses:
                continue
            
            lines1 = lines_2d[frame_id1]
            lines2 = lines_2d[frame_id2]
            pose1 = camera_poses[frame_id1]
            pose2 = camera_poses[frame_id2]
            
            # Find matches
            matches = self.match_lines_across_views(lines1, lines2, pose1, pose2)
            
            # Triangulate matched lines
            for idx1, idx2 in matches:
                line_3d = self.triangulate_line(
                    lines1[idx1],
                    lines2[idx2],
                    pose1,
                    pose2
                )
                
                if line_3d is not None:
                    lines_3d.append(line_3d)
        
        # Merge duplicate 3D lines
        lines_3d = self._merge_duplicate_lines(lines_3d)
        
        # Filter by minimum support views
        lines_3d = [line for line in lines_3d 
                   if len(line.support_views) >= self.min_support_views]
        
        return lines_3d
    
    def _merge_duplicate_lines(self, lines_3d: List[Line3D]) -> List[Line3D]:
        """
        Merge 3D lines that are likely duplicates.
        
        Args:
            lines_3d: List of 3D lines
            
        Returns:
            Merged list of 3D lines
        """
        if len(lines_3d) == 0:
            return []
        
        # Simple merging based on spatial proximity
        merged = []
        used = np.zeros(len(lines_3d), dtype=bool)
        
        for i, line1 in enumerate(lines_3d):
            if used[i]:
                continue
            
            # Find similar lines
            similar_indices = [i]
            
            for j, line2 in enumerate(lines_3d[i+1:], start=i+1):
                if used[j]:
                    continue
                
                # Check if lines are similar
                dist_start = np.linalg.norm(line1.start - line2.start)
                dist_end = np.linalg.norm(line1.end - line2.end)
                
                if dist_start < 0.1 and dist_end < 0.1:  # Similar lines
                    similar_indices.append(j)
                    used[j] = True
            
            # Merge similar lines by averaging
            if len(similar_indices) > 0:
                starts = np.array([lines_3d[idx].start for idx in similar_indices])
                ends = np.array([lines_3d[idx].end for idx in similar_indices])
                scores = np.array([lines_3d[idx].score for idx in similar_indices])
                
                merged_start = np.average(starts, axis=0, weights=scores)
                merged_end = np.average(ends, axis=0, weights=scores)
                merged_score = np.mean(scores)
                
                # Combine support views
                support_views = []
                for idx in similar_indices:
                    support_views.extend(lines_3d[idx].support_views)
                support_views = list(set(support_views))
                
                merged_line = Line3D(
                    merged_start,
                    merged_end,
                    merged_score,
                    support_views
                )
                merged.append(merged_line)
            
            used[i] = True
        
        return merged
