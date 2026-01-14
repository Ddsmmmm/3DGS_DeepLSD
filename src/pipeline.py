"""
Main pipeline for 3DGS with DeepLSD line constraints.

This module provides the end-to-end pipeline for:
1. Detecting 2D lines from video frames using DeepLSD
2. Reconstructing 3D lines from 2D detections
3. Constraining 3D Gaussian ellipsoids using the reconstructed lines
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from tqdm import tqdm

from line_detection import LSDLineDetector, VideoFrameLoader
from line_reconstruction import (
    LineReconstructor, Line2D, Line3D, CameraPose,
    load_colmap_cameras, load_transforms_json, align_poses_to_3dgs
)
from gaussian_constraint import (
    GaussianConstraintLoss, GaussianLineInitializer, GaussianParams
)
from utils import (
    draw_lines_on_image, visualize_3d_lines,
    visualize_gaussians_and_lines, plot_line_statistics
)


class Pipeline3DGS_DeepLSD:
    """
    Main pipeline for 3D Gaussian Splatting with DeepLSD line constraints.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize components
        self.line_detector = None
        self.line_reconstructor = None
        self.gaussian_constraint = None
        self.gaussian_initializer = None
        
        self._setup_components()
    
    def _setup_components(self):
        """Setup pipeline components from config."""
        # Line detector
        detector_config = self.config.get('line_detector', {})
        detector_type = detector_config.get('type', 'lsd')
        
        if detector_type == 'lsd':
            self.line_detector = LSDLineDetector(
                min_length=detector_config.get('min_length', 15.0),
                max_num_lines=detector_config.get('max_num_lines', 500)
            )
        else:
            raise ValueError(f"Unsupported detector type: {detector_type}")
        
        # Line reconstructor
        reconstructor_config = self.config.get('line_reconstructor', {})
        self.line_reconstructor = LineReconstructor(
            reprojection_threshold=reconstructor_config.get('reprojection_threshold', 5.0),
            min_triangulation_angle=reconstructor_config.get('min_triangulation_angle', 5.0),
            min_support_views=reconstructor_config.get('min_support_views', 2)
        )
        
        # Gaussian constraint
        constraint_config = self.config.get('gaussian_constraint', {})
        self.gaussian_constraint = GaussianConstraintLoss(
            line_weight=constraint_config.get('line_weight', 1.0),
            density_weight=constraint_config.get('density_weight', 0.5),
            alignment_weight=constraint_config.get('alignment_weight', 0.3),
            max_distance=constraint_config.get('max_distance', 0.1)
        )
        
        # Gaussian initializer
        initializer_config = self.config.get('gaussian_initializer', {})
        self.gaussian_initializer = GaussianLineInitializer(
            num_gaussians_per_line=initializer_config.get('num_gaussians_per_line', 10),
            initial_scale=initializer_config.get('initial_scale', 0.01),
            initial_opacity=initializer_config.get('initial_opacity', 0.5)
        )
    
    def detect_lines_from_video(self,
                               video_path: Optional[str] = None,
                               image_dir: Optional[str] = None,
                               output_dir: Optional[str] = None) -> Dict[int, List[Line2D]]:
        """
        Detect 2D lines from video frames.
        
        Args:
            video_path: Path to video file
            image_dir: Path to image directory
            output_dir: Optional directory to save visualizations
            
        Returns:
            Dictionary mapping frame_id to list of Line2D objects
        """
        print("Detecting 2D lines from video frames...")
        
        # Setup frame loader
        frame_config = self.config.get('video_loader', {})
        loader = VideoFrameLoader(
            video_path=video_path,
            image_dir=image_dir,
            target_size=frame_config.get('target_size'),
            frame_skip=frame_config.get('frame_skip', 1)
        )
        
        lines_2d = {}
        frames_for_viz = []
        lines_for_viz = []
        
        # Process each frame
        for frame_id, frame in tqdm(loader.load_frames(), desc="Detecting lines"):
            # Detect lines
            lines, scores = self.line_detector.detect_lines(frame)
            
            # Convert to Line2D objects
            line_objects = []
            for line, score in zip(lines, scores):
                line_objects.append(Line2D(line, frame_id, score))
            
            lines_2d[frame_id] = line_objects
            
            # Store for visualization
            if output_dir is not None and frame_id % 10 == 0:
                frames_for_viz.append(frame)
                lines_for_viz.append(lines)
        
        print(f"Detected lines in {len(lines_2d)} frames")
        
        # Save visualizations
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save sample frames with lines
            for i, (frame, lines) in enumerate(zip(frames_for_viz[:5], lines_for_viz[:5])):
                viz_frame = draw_lines_on_image(frame, lines)
                import cv2
                cv2.imwrite(str(output_path / f'lines_frame_{i}.jpg'), viz_frame)
            
            # Plot statistics
            plot_line_statistics(
                lines_for_viz,
                np.array([]),
                save_path=str(output_path / 'line_statistics.png')
            )
        
        return lines_2d
    
    def reconstruct_3d_lines(self,
                            lines_2d: Dict[int, List[Line2D]],
                            camera_poses: Dict[int, CameraPose],
                            output_dir: Optional[str] = None) -> List[Line3D]:
        """
        Reconstruct 3D lines from 2D detections.
        
        Args:
            lines_2d: Dictionary of 2D lines per frame
            camera_poses: Camera poses for each frame
            output_dir: Optional directory to save visualizations
            
        Returns:
            List of reconstructed 3D lines
        """
        print("Reconstructing 3D lines...")
        
        # Reconstruct lines
        lines_3d = self.line_reconstructor.reconstruct_lines(lines_2d, camera_poses)
        
        print(f"Reconstructed {len(lines_3d)} 3D lines")
        
        # Save visualizations
        if output_dir is not None and len(lines_3d) > 0:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Extract line endpoints
            line_starts = np.array([line.start for line in lines_3d])
            line_ends = np.array([line.end for line in lines_3d])
            
            # Get camera centers
            camera_centers = np.array([pose.C.flatten() for pose in camera_poses.values()])
            
            # Visualize 3D lines
            visualize_3d_lines(
                line_starts,
                line_ends,
                camera_centers,
                save_path=str(output_path / '3d_lines.png')
            )
            
            # Save as JSON
            lines_data = []
            for line in lines_3d:
                lines_data.append({
                    'start': line.start.tolist(),
                    'end': line.end.tolist(),
                    'score': float(line.score),
                    'support_views': line.support_views
                })
            
            with open(output_path / '3d_lines.json', 'w') as f:
                json.dump(lines_data, f, indent=2)
        
        return lines_3d
    
    def initialize_gaussians_on_lines(self,
                                     lines_3d: List[Line3D],
                                     output_dir: Optional[str] = None) -> List[GaussianParams]:
        """
        Initialize Gaussian ellipsoids along 3D lines.
        
        Args:
            lines_3d: List of 3D lines
            output_dir: Optional directory to save visualizations
            
        Returns:
            List of initialized Gaussian parameters
        """
        print("Initializing Gaussians on 3D lines...")
        
        # Extract line data
        line_starts = np.array([line.start for line in lines_3d])
        line_ends = np.array([line.end for line in lines_3d])
        line_scores = np.array([line.score for line in lines_3d])
        
        # Initialize Gaussians
        gaussians = self.gaussian_initializer.initialize_gaussians_on_lines(
            line_starts, line_ends, line_scores
        )
        
        print(f"Initialized {len(gaussians)} Gaussians")
        
        # Save visualizations
        if output_dir is not None and len(gaussians) > 0:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Extract Gaussian positions and scales
            gaussian_positions = np.array([g.position for g in gaussians])
            gaussian_scales = np.array([g.scale for g in gaussians])
            
            # Visualize
            visualize_gaussians_and_lines(
                gaussian_positions,
                gaussian_scales,
                line_starts,
                line_ends,
                save_path=str(output_path / 'gaussians_and_lines.ply')
            )
            
            # Save Gaussian parameters
            gaussians_data = []
            for g in gaussians:
                gaussians_data.append({
                    'position': g.position.tolist(),
                    'rotation': g.rotation.tolist(),
                    'scale': g.scale.tolist(),
                    'opacity': float(g.opacity),
                    'color': g.color.tolist()
                })
            
            with open(output_path / 'gaussians.json', 'w') as f:
                json.dump(gaussians_data, f, indent=2)
        
        return gaussians
    
    def run_full_pipeline(self,
                         video_path: Optional[str] = None,
                         image_dir: Optional[str] = None,
                         camera_file: Optional[str] = None,
                         output_dir: str = './output') -> Dict:
        """
        Run the full pipeline from video to constrained Gaussians.
        
        Args:
            video_path: Path to video file
            image_dir: Path to image directory
            camera_file: Path to camera poses file (COLMAP or transforms.json)
            output_dir: Output directory for results
            
        Returns:
            Dictionary with results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Detect 2D lines
        lines_2d = self.detect_lines_from_video(
            video_path=video_path,
            image_dir=image_dir,
            output_dir=output_dir
        )
        
        # Step 2: Load camera poses
        print("Loading camera poses...")
        if camera_file is None:
            raise ValueError("Camera poses file must be provided")
        
        if 'transforms.json' in camera_file:
            camera_poses = load_transforms_json(camera_file)
        else:
            # Assume COLMAP format
            cameras_file = Path(camera_file).parent / 'cameras.txt'
            images_file = Path(camera_file).parent / 'images.txt'
            camera_poses = load_colmap_cameras(str(cameras_file), str(images_file))
        
        # Align to 3DGS coordinate system
        align_config = self.config.get('alignment', {})
        if align_config.get('enable', True):
            camera_poses = align_poses_to_3dgs(
                camera_poses,
                reference_point=align_config.get('reference_point'),
                reference_scale=align_config.get('reference_scale', 1.0)
            )
        
        # Step 3: Reconstruct 3D lines
        lines_3d = self.reconstruct_3d_lines(
            lines_2d,
            camera_poses,
            output_dir=output_dir
        )
        
        # Step 4: Initialize Gaussians on lines
        gaussians = self.initialize_gaussians_on_lines(
            lines_3d,
            output_dir=output_dir
        )
        
        print(f"\nPipeline complete! Results saved to {output_dir}")
        
        return {
            'lines_2d': lines_2d,
            'lines_3d': lines_3d,
            'gaussians': gaussians,
            'camera_poses': camera_poses
        }
