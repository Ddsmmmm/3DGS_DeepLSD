"""
Camera pose estimation and loading utilities.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .line_reconstructor import CameraPose


def load_colmap_cameras(cameras_file: str,
                        images_file: str) -> Dict[int, CameraPose]:
    """
    Load camera poses from COLMAP text format.
    
    Args:
        cameras_file: Path to cameras.txt file
        images_file: Path to images.txt file
        
    Returns:
        Dictionary mapping frame_id to CameraPose
    """
    # Parse cameras.txt to get intrinsics
    cameras_intrinsics = {}
    
    with open(cameras_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            
            # Assuming PINHOLE model: fx, fy, cx, cy
            if model == 'PINHOLE' and len(params) >= 4:
                fx, fy, cx, cy = params[:4]
                K = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ])
                cameras_intrinsics[camera_id] = K
    
    # Parse images.txt to get extrinsics
    camera_poses = {}
    
    with open(images_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        # Images.txt has pairs of lines: image info and points
        for i in range(0, len(lines), 2):
            parts = lines[i].split()
            
            image_id = int(parts[0])
            qw, qx, qy, qz = [float(p) for p in parts[1:5]]
            tx, ty, tz = [float(p) for p in parts[5:8]]
            camera_id = int(parts[8])
            
            if camera_id not in cameras_intrinsics:
                continue
            
            # Convert quaternion to rotation matrix
            # COLMAP uses quaternion as [w, x, y, z]
            from scipy.spatial.transform import Rotation
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            t = np.array([tx, ty, tz])
            
            K = cameras_intrinsics[camera_id]
            
            camera_poses[image_id] = CameraPose(K, R, t, image_id)
    
    return camera_poses


def load_transforms_json(transforms_file: str) -> Dict[int, CameraPose]:
    """
    Load camera poses from transforms.json (NeRF/3DGS format).
    
    Args:
        transforms_file: Path to transforms.json file
        
    Returns:
        Dictionary mapping frame_id to CameraPose
    """
    with open(transforms_file, 'r') as f:
        data = json.load(f)
    
    camera_poses = {}
    
    # Get camera intrinsics
    if 'camera_angle_x' in data:
        angle_x = data['camera_angle_x']
        w = data.get('w', 800)
        h = data.get('h', 800)
        fx = w / (2 * np.tan(angle_x / 2))
        fy = fx
        cx = w / 2
        cy = h / 2
    elif 'fl_x' in data:
        fx = data['fl_x']
        fy = data.get('fl_y', fx)
        cx = data.get('cx', data.get('w', 800) / 2)
        cy = data.get('cy', data.get('h', 800) / 2)
    else:
        # Default intrinsics
        fx = fy = 800
        cx = cy = 400
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    # Load frames
    for frame_id, frame in enumerate(data['frames']):
        # Transform matrix is camera-to-world (4x4)
        transform_matrix = np.array(frame['transform_matrix'])
        
        # Extract rotation and translation (world-to-camera)
        c2w = transform_matrix[:3, :4]
        R_w2c = c2w[:3, :3].T
        t_w2c = -R_w2c @ c2w[:3, 3]
        
        camera_poses[frame_id] = CameraPose(K, R_w2c, t_w2c, frame_id)
    
    return camera_poses


def estimate_camera_poses_from_sfm(image_dir: str,
                                   output_dir: str) -> Dict[int, CameraPose]:
    """
    Estimate camera poses using Structure-from-Motion.
    
    This is a placeholder that would integrate with COLMAP or similar SfM system.
    
    Args:
        image_dir: Directory containing images
        output_dir: Directory to save SfM output
        
    Returns:
        Dictionary mapping frame_id to CameraPose
    """
    # This would call COLMAP or similar SfM pipeline
    # For now, return empty dict as placeholder
    print("Warning: SfM estimation not implemented. Please provide camera poses.")
    return {}


def align_poses_to_3dgs(camera_poses: Dict[int, CameraPose],
                        reference_point: Optional[np.ndarray] = None,
                        reference_scale: float = 1.0) -> Dict[int, CameraPose]:
    """
    Align camera poses to 3DGS coordinate system.
    
    Args:
        camera_poses: Original camera poses
        reference_point: Reference point for alignment (origin)
        reference_scale: Scale factor for alignment
        
    Returns:
        Aligned camera poses
    """
    if reference_point is None:
        # Compute scene center from camera positions
        camera_centers = np.array([pose.C.flatten() for pose in camera_poses.values()])
        reference_point = np.mean(camera_centers, axis=0)
    
    aligned_poses = {}
    
    for frame_id, pose in camera_poses.items():
        # Translate camera center
        t_aligned = pose.t - pose.R @ reference_point.reshape(3, 1)
        
        # Apply scale
        t_aligned = t_aligned * reference_scale
        
        # Create aligned pose
        aligned_poses[frame_id] = CameraPose(
            pose.K,
            pose.R,
            t_aligned,
            frame_id
        )
    
    return aligned_poses
