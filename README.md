# 3DGS_DeepLSD

A 3D reconstruction method that integrates DeepLSD line detection with 3D Gaussian Splatting (3DGS) to improve model quality in less-ideally sampled directions.

## Overview

This project extracts 2D line features from video frames using line detection algorithms (DeepLSD/LSD), reconstructs 3D lines by triangulating matched lines across multiple views, and uses these 3D lines to constrain the distribution of 3D Gaussian ellipsoids in the 3DGS framework. This approach improves the quality and completeness of 3D reconstruction, particularly in regions with limited viewpoint coverage.

## Key Features

- **2D Line Detection**: Extract line features from video frames using LSD (Line Segment Detector) or DeepLSD
- **3D Line Reconstruction**: Triangulate 2D lines across multiple views to reconstruct 3D line segments
- **Coordinate Alignment**: Automatically align reconstructed 3D lines with the 3DGS coordinate system
- **Gaussian Constraints**: Use 3D lines to constrain and guide the placement of 3D Gaussian ellipsoids
- **Initialization**: Initialize Gaussians along 3D lines for better coverage
- **Visualization**: Comprehensive visualization tools for lines and Gaussians

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for DeepLSD)
- Camera poses in COLMAP format or transforms.json (NeRF/3DGS format)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Ddsmmmm/3DGS_DeepLSD.git
cd 3DGS_DeepLSD
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package:
```bash
pip install -e .
```

## Usage

### Basic Usage

Process a video with camera poses:

```bash
python src/main.py \
    --video path/to/video.mp4 \
    --camera-file path/to/transforms.json \
    --output ./output
```

Process a directory of images:

```bash
python src/main.py \
    --images path/to/images/ \
    --camera-file path/to/colmap/directory \
    --output ./output
```

### Advanced Usage

Use a configuration file for fine-tuned control:

```bash
python src/main.py \
    --video path/to/video.mp4 \
    --camera-file path/to/transforms.json \
    --config configs/high_quality_config.yaml \
    --output ./output
```

### Command-line Options

- `--video`: Path to input video file
- `--images`: Path to directory containing input images
- `--camera-file`: Path to camera poses file (transforms.json or COLMAP directory)
- `--output`: Output directory for results (default: ./output)
- `--config`: Path to configuration YAML file
- `--detector`: Line detector to use: 'lsd' or 'deeplsd' (default: lsd)
- `--min-line-length`: Minimum line length in pixels (default: 15.0)
- `--max-lines`: Maximum number of lines per frame (default: 500)
- `--frame-skip`: Process every Nth frame (default: 1)

## Configuration

Three pre-configured settings are provided:

1. **default_config.yaml**: Balanced quality and speed
2. **high_quality_config.yaml**: Maximum quality, slower processing
3. **fast_config.yaml**: Faster processing, lower quality

### Configuration Parameters

#### Line Detector
- `type`: Detector type ('lsd' or 'deeplsd')
- `min_length`: Minimum line length in pixels
- `max_num_lines`: Maximum lines per frame
- `score_threshold`: Confidence threshold (for DeepLSD)

#### Line Reconstructor
- `reprojection_threshold`: Maximum reprojection error in pixels
- `min_triangulation_angle`: Minimum angle between views (degrees)
- `min_support_views`: Minimum views supporting a 3D line

#### Gaussian Constraint
- `line_weight`: Weight for line proximity loss
- `density_weight`: Weight for density regularization
- `alignment_weight`: Weight for orientation alignment
- `max_distance`: Maximum distance for line-Gaussian association

#### Gaussian Initializer
- `num_gaussians_per_line`: Gaussians per 3D line
- `initial_scale`: Initial scale for Gaussians
- `initial_opacity`: Initial opacity value

## Pipeline Architecture

The pipeline consists of four main stages:

### 1. 2D Line Detection
Extracts line segments from each video frame using:
- **LSD (Line Segment Detector)**: Fast, traditional method
- **DeepLSD**: Deep learning-based method (not yet fully integrated)

### 2. 3D Line Reconstruction
Reconstructs 3D lines by:
- Matching lines across multiple views using epipolar geometry
- Triangulating matched 2D lines to 3D space
- Filtering based on reprojection error and triangulation angle
- Merging duplicate 3D lines

### 3. Coordinate Alignment
Aligns 3D lines with the 3DGS coordinate system:
- Centers the scene at a reference point
- Applies scale normalization
- Ensures consistency with existing 3DGS reconstructions

### 4. Gaussian Constraint
Improves 3DGS quality by:
- Initializing Gaussians along 3D lines
- Applying line proximity loss
- Regularizing density along lines
- Aligning Gaussian orientations with lines

## Output

The pipeline generates several outputs in the specified output directory:

- `lines_frame_*.jpg`: Sample frames with detected 2D lines visualized
- `line_statistics.png`: Statistics about detected and reconstructed lines
- `3d_lines.png`: Visualization of reconstructed 3D lines
- `3d_lines.json`: 3D line data in JSON format
- `gaussians_and_lines.ply`: Point cloud visualization
- `gaussians.json`: Initialized Gaussian parameters

## Integration with 3DGS

The output Gaussians can be integrated into existing 3DGS frameworks:

1. Load the initialized Gaussians from `gaussians.json`
2. Use the constraint loss during 3DGS training:
   ```python
   from gaussian_constraint import GaussianConstraintLoss
   
   constraint_loss = GaussianConstraintLoss(...)
   loss, loss_dict = constraint_loss(
       gaussian_positions,
       gaussian_rotations,
       gaussian_scales,
       gaussian_opacities,
       line_starts,
       line_ends
   )
   ```
3. Add the constraint loss to the total training loss

## Camera Pose Formats

### COLMAP Format
Requires `cameras.txt` and `images.txt`:
```bash
--camera-file path/to/colmap/directory
```

### transforms.json (NeRF/3DGS Format)
```json
{
  "camera_angle_x": 0.8575560450553894,
  "frames": [
    {
      "file_path": "./images/frame_00000.jpg",
      "transform_matrix": [[...], [...], [...], [...]]
    },
    ...
  ]
}
```

## Examples

### Process a video with high quality settings:
```bash
python src/main.py \
    --video data/video.mp4 \
    --camera-file data/transforms.json \
    --config configs/high_quality_config.yaml \
    --output results/high_quality
```

### Fast processing for quick preview:
```bash
python src/main.py \
    --images data/images/ \
    --camera-file data/colmap/ \
    --config configs/fast_config.yaml \
    --frame-skip 10 \
    --output results/preview
```

## Project Structure

```
3DGS_DeepLSD/
├── src/
│   ├── line_detection/         # 2D line detection modules
│   │   ├── line_detector.py    # DeepLSD and LSD detectors
│   │   └── video_loader.py     # Frame loading utilities
│   ├── line_reconstruction/    # 3D line reconstruction
│   │   ├── line_reconstructor.py  # Triangulation and matching
│   │   └── camera_utils.py     # Camera pose utilities
│   ├── gaussian_constraint/    # Gaussian constraint system
│   │   └── gaussian_constraint.py  # Loss functions and initialization
│   ├── utils/                  # Utilities
│   │   └── visualization.py    # Visualization tools
│   ├── pipeline.py             # Main pipeline
│   └── main.py                 # CLI entry point
├── configs/                    # Configuration files
│   ├── default_config.yaml
│   ├── high_quality_config.yaml
│   └── fast_config.yaml
├── requirements.txt            # Dependencies
├── setup.py                    # Package setup
└── README.md                   # This file
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{3dgs_deeplsd,
  title={3DGS\_DeepLSD: Improving 3D Gaussian Splatting with Line Constraints},
  author={3DGS\_DeepLSD Contributors},
  year={2024},
  url={https://github.com/Ddsmmmm/3DGS_DeepLSD}
}
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) for the base 3DGS framework
- [DeepLSD](https://github.com/cvg/DeepLSD) for the line detection method
- [LSD](https://www.ipol.im/pub/art/2012/gjmr-lsd/) for the fast line segment detector

## Contact

For questions or issues, please open an issue on GitHub.