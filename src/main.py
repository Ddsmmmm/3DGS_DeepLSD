"""
Command-line interface for 3DGS_DeepLSD pipeline.
"""

import argparse
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import Pipeline3DGS_DeepLSD


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description='3DGS with DeepLSD Line Constraints Pipeline'
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--video',
        type=str,
        help='Path to input video file'
    )
    input_group.add_argument(
        '--images',
        type=str,
        help='Path to directory containing input images'
    )
    
    # Camera poses
    parser.add_argument(
        '--camera-file',
        type=str,
        required=True,
        help='Path to camera poses file (transforms.json or COLMAP directory)'
    )
    
    # Output
    parser.add_argument(
        '--output',
        type=str,
        default='./output',
        help='Output directory for results (default: ./output)'
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration YAML file'
    )
    
    # Pipeline options
    parser.add_argument(
        '--detector',
        type=str,
        default='lsd',
        choices=['lsd', 'deeplsd'],
        help='Line detector to use (default: lsd)'
    )
    
    parser.add_argument(
        '--min-line-length',
        type=float,
        default=15.0,
        help='Minimum line length in pixels (default: 15.0)'
    )
    
    parser.add_argument(
        '--max-lines',
        type=int,
        default=500,
        help='Maximum number of lines per frame (default: 500)'
    )
    
    parser.add_argument(
        '--frame-skip',
        type=int,
        default=1,
        help='Process every Nth frame (default: 1)'
    )
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        config = load_config(args.config)
    else:
        # Create default config
        config = {
            'line_detector': {
                'type': args.detector,
                'min_length': args.min_line_length,
                'max_num_lines': args.max_lines
            },
            'video_loader': {
                'frame_skip': args.frame_skip,
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
    print("=" * 60)
    print("3DGS with DeepLSD Line Constraints")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Input: {args.video or args.images}")
    print(f"  Camera file: {args.camera_file}")
    print(f"  Output directory: {args.output}")
    print(f"  Line detector: {config['line_detector']['type']}")
    print(f"  Min line length: {config['line_detector']['min_length']} px")
    print(f"  Max lines per frame: {config['line_detector']['max_num_lines']}")
    print()
    
    pipeline = Pipeline3DGS_DeepLSD(config)
    
    # Run pipeline
    try:
        results = pipeline.run_full_pipeline(
            video_path=args.video,
            image_dir=args.images,
            camera_file=args.camera_file,
            output_dir=args.output
        )
        
        print("\n" + "=" * 60)
        print("Pipeline Results:")
        print("=" * 60)
        print(f"  2D lines detected: {sum(len(lines) for lines in results['lines_2d'].values())}")
        print(f"  3D lines reconstructed: {len(results['lines_3d'])}")
        print(f"  Gaussians initialized: {len(results['gaussians'])}")
        print(f"\nResults saved to: {args.output}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
