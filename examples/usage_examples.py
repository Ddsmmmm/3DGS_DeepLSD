"""
Example script demonstrating how to use the 3DGS_DeepLSD pipeline programmatically.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pipeline import Pipeline3DGS_DeepLSD
import numpy as np


def example_basic_usage():
    """Basic usage example."""
    
    # Define configuration
    config = {
        'line_detector': {
            'type': 'lsd',
            'min_length': 15.0,
            'max_num_lines': 500
        },
        'video_loader': {
            'frame_skip': 1,
            'target_size': None
        },
        'line_reconstructor': {
            'reprojection_threshold': 5.0,
            'min_triangulation_angle': 5.0,
            'min_support_views': 2
        },
        'gaussian_constraint': {
            'line_weight': 1.0,
            'density_weight': 0.5,
            'alignment_weight': 0.3,
            'max_distance': 0.1
        },
        'gaussian_initializer': {
            'num_gaussians_per_line': 10,
            'initial_scale': 0.01,
            'initial_opacity': 0.5
        },
        'alignment': {
            'enable': True,
            'reference_point': None,
            'reference_scale': 1.0
        }
    }
    
    # Initialize pipeline
    pipeline = Pipeline3DGS_DeepLSD(config)
    
    # Run full pipeline
    # Note: Replace these paths with actual data
    results = pipeline.run_full_pipeline(
        video_path='data/video.mp4',  # or image_dir='data/images/'
        camera_file='data/transforms.json',  # or COLMAP directory
        output_dir='output/example'
    )
    
    # Access results
    print(f"Detected {len(results['lines_2d'])} frames with 2D lines")
    print(f"Reconstructed {len(results['lines_3d'])} 3D lines")
    print(f"Initialized {len(results['gaussians'])} Gaussians")
    
    return results


def example_step_by_step():
    """Step-by-step usage example with more control."""
    
    config = {
        'line_detector': {'type': 'lsd', 'min_length': 20.0, 'max_num_lines': 300},
        'video_loader': {'frame_skip': 2},
        'line_reconstructor': {'reprojection_threshold': 4.0, 'min_support_views': 2},
        'gaussian_constraint': {'line_weight': 1.0},
        'gaussian_initializer': {'num_gaussians_per_line': 15},
        'alignment': {'enable': True}
    }
    
    pipeline = Pipeline3DGS_DeepLSD(config)
    
    # Step 1: Detect 2D lines
    lines_2d = pipeline.detect_lines_from_video(
        image_dir='data/images/',
        output_dir='output/step_by_step'
    )
    
    # Step 2: Load camera poses
    from line_reconstruction import load_transforms_json
    camera_poses = load_transforms_json('data/transforms.json')
    
    # Step 3: Reconstruct 3D lines
    lines_3d = pipeline.reconstruct_3d_lines(
        lines_2d,
        camera_poses,
        output_dir='output/step_by_step'
    )
    
    # Step 4: Initialize Gaussians
    gaussians = pipeline.initialize_gaussians_on_lines(
        lines_3d,
        output_dir='output/step_by_step'
    )
    
    return lines_2d, lines_3d, gaussians


def example_constraint_loss():
    """Example of using the Gaussian constraint loss in training."""
    
    import torch
    from gaussian_constraint import GaussianConstraintLoss
    
    # Initialize loss
    constraint_loss = GaussianConstraintLoss(
        line_weight=1.0,
        density_weight=0.5,
        alignment_weight=0.3,
        max_distance=0.1
    )
    
    # Example Gaussian parameters (in practice, these come from your 3DGS model)
    num_gaussians = 1000
    gaussian_positions = torch.randn(num_gaussians, 3, requires_grad=True)
    gaussian_rotations = torch.randn(num_gaussians, 4, requires_grad=True)
    gaussian_scales = torch.rand(num_gaussians, 3, requires_grad=True) * 0.1
    gaussian_opacities = torch.rand(num_gaussians, requires_grad=True)
    
    # Example 3D lines (in practice, these come from reconstruction)
    num_lines = 100
    line_starts = torch.randn(num_lines, 3)
    line_ends = line_starts + torch.randn(num_lines, 3) * 0.5
    
    # Compute loss
    loss, loss_dict = constraint_loss(
        gaussian_positions,
        gaussian_rotations,
        gaussian_scales,
        gaussian_opacities,
        line_starts,
        line_ends
    )
    
    print(f"Total loss: {loss.item():.4f}")
    print(f"Proximity loss: {loss_dict['proximity'].item():.4f}")
    print(f"Density loss: {loss_dict['density'].item():.4f}")
    print(f"Alignment loss: {loss_dict['alignment'].item():.4f}")
    
    # Use in optimization
    loss.backward()
    
    return loss, loss_dict


if __name__ == '__main__':
    print("=" * 60)
    print("3DGS_DeepLSD Usage Examples")
    print("=" * 60)
    
    print("\nExample 1: Basic Usage")
    print("-" * 60)
    print("Demonstrates the simplest way to run the full pipeline.")
    print("Uncomment the line below to run:")
    print("# results = example_basic_usage()")
    
    print("\nExample 2: Step-by-Step")
    print("-" * 60)
    print("Demonstrates running each pipeline stage separately for more control.")
    print("Uncomment the line below to run:")
    print("# lines_2d, lines_3d, gaussians = example_step_by_step()")
    
    print("\nExample 3: Constraint Loss")
    print("-" * 60)
    print("Demonstrates how to use the Gaussian constraint loss in training.")
    print("This example can run without data:")
    loss, loss_dict = example_constraint_loss()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
