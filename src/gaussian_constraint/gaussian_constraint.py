"""
3D Gaussian Constraint Module

This module implements constraints on 3D Gaussian ellipsoids using reconstructed 3D lines
to improve 3DGS model quality in less-ideally sampled directions.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class GaussianParams:
    """Parameters for a 3D Gaussian ellipsoid."""
    position: np.ndarray  # (3,) - mean position
    rotation: np.ndarray  # (4,) - quaternion rotation
    scale: np.ndarray     # (3,) - scale in 3 axes
    opacity: float        # Opacity value
    color: np.ndarray     # (3,) or (N, 3) - RGB or spherical harmonics


class GaussianConstraintLoss(nn.Module):
    """
    Loss function to constrain Gaussian ellipsoids using 3D line information.
    
    This loss encourages Gaussians to align with detected 3D lines, improving
    reconstruction quality in less-ideally sampled directions.
    """
    
    def __init__(self,
                 line_weight: float = 1.0,
                 density_weight: float = 0.5,
                 alignment_weight: float = 0.3,
                 max_distance: float = 0.1):
        """
        Initialize constraint loss.
        
        Args:
            line_weight: Weight for line proximity loss
            density_weight: Weight for density regularization along lines
            alignment_weight: Weight for orientation alignment loss
            max_distance: Maximum distance for line-Gaussian association
        """
        super().__init__()
        self.line_weight = line_weight
        self.density_weight = density_weight
        self.alignment_weight = alignment_weight
        self.max_distance = max_distance
    
    def compute_point_to_line_distance(self,
                                      points: torch.Tensor,
                                      line_start: torch.Tensor,
                                      line_end: torch.Tensor) -> torch.Tensor:
        """
        Compute distance from points to line segments.
        
        Args:
            points: Points (N, 3)
            line_start: Line start points (M, 3)
            line_end: Line end points (M, 3)
            
        Returns:
            Distance matrix (N, M)
        """
        # Broadcast to compute pairwise distances
        # points: (N, 1, 3), line_start: (1, M, 3)
        points = points.unsqueeze(1)
        line_start = line_start.unsqueeze(0)
        line_end = line_end.unsqueeze(0)
        
        # Line direction and length
        line_dir = line_end - line_start
        line_length_sq = torch.sum(line_dir ** 2, dim=-1, keepdim=True)
        line_length_sq = torch.clamp(line_length_sq, min=1e-8)
        
        # Project points onto lines
        t = torch.sum((points - line_start) * line_dir, dim=-1, keepdim=True) / line_length_sq
        t = torch.clamp(t, 0, 1)
        
        # Closest point on line
        closest_point = line_start + t * line_dir
        
        # Distance to line
        distances = torch.norm(points - closest_point, dim=-1)
        
        return distances
    
    def line_proximity_loss(self,
                           gaussian_positions: torch.Tensor,
                           line_starts: torch.Tensor,
                           line_ends: torch.Tensor) -> torch.Tensor:
        """
        Compute loss encouraging Gaussians to be near lines.
        
        Args:
            gaussian_positions: Gaussian centers (N, 3)
            line_starts: Line start points (M, 3)
            line_ends: Line end points (M, 3)
            
        Returns:
            Loss value
        """
        if len(line_starts) == 0:
            return torch.tensor(0.0, device=gaussian_positions.device)
        
        # Compute distances to all lines
        distances = self.compute_point_to_line_distance(
            gaussian_positions, line_starts, line_ends
        )
        
        # Find minimum distance to any line for each Gaussian
        min_distances, _ = torch.min(distances, dim=1)
        
        # Soft constraint: penalize Gaussians far from lines
        loss = torch.mean(torch.exp(-min_distances / self.max_distance) * min_distances)
        
        return loss
    
    def density_regularization_loss(self,
                                   gaussian_positions: torch.Tensor,
                                   gaussian_opacities: torch.Tensor,
                                   line_starts: torch.Tensor,
                                   line_ends: torch.Tensor,
                                   num_samples: int = 50) -> torch.Tensor:
        """
        Encourage uniform density of Gaussians along lines.
        
        Args:
            gaussian_positions: Gaussian centers (N, 3)
            gaussian_opacities: Gaussian opacities (N,)
            line_starts: Line start points (M, 3)
            line_ends: Line end points (M, 3)
            num_samples: Number of samples per line
            
        Returns:
            Loss value
        """
        if len(line_starts) == 0:
            return torch.tensor(0.0, device=gaussian_positions.device)
        
        # Sample points along each line
        t = torch.linspace(0, 1, num_samples, device=line_starts.device)
        t = t.view(1, num_samples, 1)
        
        line_starts_expanded = line_starts.unsqueeze(1)  # (M, 1, 3)
        line_ends_expanded = line_ends.unsqueeze(1)      # (M, 1, 3)
        
        samples = line_starts_expanded + t * (line_ends_expanded - line_starts_expanded)
        samples = samples.view(-1, 3)  # (M * num_samples, 3)
        
        # Find nearby Gaussians for each sample
        distances = torch.cdist(samples, gaussian_positions)  # (M * num_samples, N)
        
        # Compute density at each sample using Gaussian influence
        weights = torch.exp(-distances ** 2 / (2 * self.max_distance ** 2))
        weighted_opacities = weights * gaussian_opacities.unsqueeze(0)
        densities = torch.sum(weighted_opacities, dim=1)
        
        # Reshape to per-line samples
        densities = densities.view(len(line_starts), num_samples)
        
        # Encourage uniform density along each line
        mean_density = torch.mean(densities, dim=1, keepdim=True)
        variance = torch.mean((densities - mean_density) ** 2, dim=1)
        
        loss = torch.mean(variance)
        
        return loss
    
    def alignment_loss(self,
                      gaussian_positions: torch.Tensor,
                      gaussian_rotations: torch.Tensor,
                      gaussian_scales: torch.Tensor,
                      line_starts: torch.Tensor,
                      line_ends: torch.Tensor) -> torch.Tensor:
        """
        Encourage Gaussian ellipsoids to align with nearby lines.
        
        Args:
            gaussian_positions: Gaussian centers (N, 3)
            gaussian_rotations: Gaussian rotations as quaternions (N, 4)
            gaussian_scales: Gaussian scales (N, 3)
            line_starts: Line start points (M, 3)
            line_ends: Line end points (M, 3)
            
        Returns:
            Loss value
        """
        if len(line_starts) == 0:
            return torch.tensor(0.0, device=gaussian_positions.device)
        
        # Find nearest line for each Gaussian
        distances = self.compute_point_to_line_distance(
            gaussian_positions, line_starts, line_ends
        )
        min_distances, nearest_line_idx = torch.min(distances, dim=1)
        
        # Only consider Gaussians close to lines
        close_mask = min_distances < self.max_distance
        
        if torch.sum(close_mask) == 0:
            return torch.tensor(0.0, device=gaussian_positions.device)
        
        # Get line directions for nearest lines
        line_directions = line_ends - line_starts
        line_directions = line_directions / (torch.norm(line_directions, dim=1, keepdim=True) + 1e-8)
        nearest_line_dirs = line_directions[nearest_line_idx[close_mask]]
        
        # Get principal axis of Gaussians (direction of largest scale)
        close_rotations = gaussian_rotations[close_mask]
        close_scales = gaussian_scales[close_mask]
        
        # Find principal axis (assuming scale order is [x, y, z])
        max_scale_idx = torch.argmax(close_scales, dim=1)
        
        # Convert quaternion to rotation matrix and extract principal axis
        # For simplicity, we'll use the first axis of the rotated frame
        principal_axes = self._quaternion_to_principal_axis(close_rotations, max_scale_idx)
        
        # Compute alignment (1 - |cos(angle)|)
        alignment = torch.abs(torch.sum(principal_axes * nearest_line_dirs, dim=1))
        misalignment = 1 - alignment
        
        loss = torch.mean(misalignment)
        
        return loss
    
    def _quaternion_to_principal_axis(self,
                                     quaternions: torch.Tensor,
                                     axis_idx: torch.Tensor) -> torch.Tensor:
        """
        Convert quaternions to rotation matrices and extract principal axes.
        
        Args:
            quaternions: Quaternions (N, 4) as [w, x, y, z]
            axis_idx: Index of principal axis for each Gaussian (N,)
            
        Returns:
            Principal axes (N, 3)
        """
        # Normalize quaternions
        quaternions = quaternions / (torch.norm(quaternions, dim=1, keepdim=True) + 1e-8)
        
        w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
        
        # Build rotation matrices
        R00 = 1 - 2 * (y**2 + z**2)
        R01 = 2 * (x*y - w*z)
        R02 = 2 * (x*z + w*y)
        R10 = 2 * (x*y + w*z)
        R11 = 1 - 2 * (x**2 + z**2)
        R12 = 2 * (y*z - w*x)
        R20 = 2 * (x*z - w*y)
        R21 = 2 * (y*z + w*x)
        R22 = 1 - 2 * (x**2 + y**2)
        
        # Stack rotation matrices
        R = torch.stack([
            torch.stack([R00, R01, R02], dim=1),
            torch.stack([R10, R11, R12], dim=1),
            torch.stack([R20, R21, R22], dim=1)
        ], dim=1)  # (N, 3, 3)
        
        # Extract principal axes based on axis_idx
        # Create one-hot encoding for axis selection
        axes = torch.eye(3, device=quaternions.device)[axis_idx]  # (N, 3)
        
        # Extract principal axis from rotation matrix
        principal_axes = torch.bmm(R, axes.unsqueeze(-1)).squeeze(-1)
        principal_axes = principal_axes / (torch.norm(principal_axes, dim=1, keepdim=True) + 1e-8)
        
        return principal_axes
    
    def forward(self,
               gaussian_positions: torch.Tensor,
               gaussian_rotations: torch.Tensor,
               gaussian_scales: torch.Tensor,
               gaussian_opacities: torch.Tensor,
               line_starts: torch.Tensor,
               line_ends: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total constraint loss.
        
        Args:
            gaussian_positions: Gaussian centers (N, 3)
            gaussian_rotations: Gaussian rotations (N, 4)
            gaussian_scales: Gaussian scales (N, 3)
            gaussian_opacities: Gaussian opacities (N,)
            line_starts: Line start points (M, 3)
            line_ends: Line end points (M, 3)
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Compute individual losses
        proximity_loss = self.line_proximity_loss(
            gaussian_positions, line_starts, line_ends
        )
        
        density_loss = self.density_regularization_loss(
            gaussian_positions, gaussian_opacities,
            line_starts, line_ends
        )
        
        align_loss = self.alignment_loss(
            gaussian_positions, gaussian_rotations, gaussian_scales,
            line_starts, line_ends
        )
        
        # Combine losses
        total_loss = (
            self.line_weight * proximity_loss +
            self.density_weight * density_loss +
            self.alignment_weight * align_loss
        )
        
        loss_dict = {
            'proximity': proximity_loss,
            'density': density_loss,
            'alignment': align_loss,
            'total': total_loss
        }
        
        return total_loss, loss_dict


class GaussianLineInitializer:
    """
    Initialize Gaussian ellipsoids along detected 3D lines.
    """
    
    def __init__(self,
                 num_gaussians_per_line: int = 10,
                 initial_scale: float = 0.01,
                 initial_opacity: float = 0.5):
        """
        Initialize Gaussian line initializer.
        
        Args:
            num_gaussians_per_line: Number of Gaussians to place per line
            initial_scale: Initial scale for Gaussians
            initial_opacity: Initial opacity value
        """
        self.num_gaussians_per_line = num_gaussians_per_line
        self.initial_scale = initial_scale
        self.initial_opacity = initial_opacity
    
    def initialize_gaussians_on_lines(self,
                                     line_starts: np.ndarray,
                                     line_ends: np.ndarray,
                                     line_scores: Optional[np.ndarray] = None) -> List[GaussianParams]:
        """
        Initialize Gaussians along 3D lines.
        
        Args:
            line_starts: Line start points (M, 3)
            line_ends: Line end points (M, 3)
            line_scores: Optional confidence scores for lines (M,)
            
        Returns:
            List of initialized Gaussian parameters
        """
        gaussians = []
        
        for i in range(len(line_starts)):
            start = line_starts[i]
            end = line_ends[i]
            
            # Sample positions along line
            t = np.linspace(0, 1, self.num_gaussians_per_line)
            positions = start[np.newaxis, :] + t[:, np.newaxis] * (end - start)[np.newaxis, :]
            
            # Compute line direction
            line_dir = end - start
            line_length = np.linalg.norm(line_dir)
            line_dir = line_dir / (line_length + 1e-8)
            
            # Create rotation to align with line
            # Align the longest axis (x) with line direction
            rotation_quat = self._align_quaternion_with_direction(line_dir)
            
            # Set scale - elongated along line direction
            scale = np.array([
                line_length / self.num_gaussians_per_line * 2,  # Along line
                self.initial_scale,  # Perpendicular
                self.initial_scale   # Perpendicular
            ])
            
            # Adjust opacity based on line score if provided
            opacity = self.initial_opacity
            if line_scores is not None:
                opacity *= line_scores[i]
            
            # Create Gaussians
            for pos in positions:
                gaussian = GaussianParams(
                    position=pos,
                    rotation=rotation_quat,
                    scale=scale,
                    opacity=opacity,
                    color=np.array([0.5, 0.5, 0.5])  # Gray default
                )
                gaussians.append(gaussian)
        
        return gaussians
    
    def _align_quaternion_with_direction(self, direction: np.ndarray) -> np.ndarray:
        """
        Create a quaternion that aligns the x-axis with the given direction.
        
        Args:
            direction: Target direction (3,)
            
        Returns:
            Quaternion (4,) as [w, x, y, z]
        """
        # Default x-axis
        x_axis = np.array([1, 0, 0])
        
        # Compute rotation axis and angle
        v = np.cross(x_axis, direction)
        s = np.linalg.norm(v)
        c = np.dot(x_axis, direction)
        
        if s < 1e-8:  # Vectors are parallel
            if c > 0:  # Same direction
                return np.array([1, 0, 0, 0])
            else:  # Opposite direction
                return np.array([0, 0, 1, 0])
        
        # Compute quaternion
        v_norm = v / s
        half_angle = np.arctan2(s, c) / 2
        
        w = np.cos(half_angle)
        xyz = v_norm * np.sin(half_angle)
        
        return np.array([w, xyz[0], xyz[1], xyz[2]])
