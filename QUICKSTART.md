# Quick Start Guide for 3DGS_DeepLSD

This guide will help you get started with 3DGS_DeepLSD quickly.

## Installation

1. **Clone the repository** (if not already done):
```bash
git clone https://github.com/Ddsmmmm/3DGS_DeepLSD.git
cd 3DGS_DeepLSD
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python examples/verify_structure.py
```

## Quick Example

### Step 1: Prepare your data

You need:
- Video file or a directory of images
- Camera poses in one of these formats:
  - `transforms.json` (NeRF/3DGS format)
  - COLMAP output (`cameras.txt` and `images.txt`)

Example directory structure:
```
data/
├── video.mp4         # OR
├── images/           # Directory with frames
│   ├── frame_0000.jpg
│   ├── frame_0001.jpg
│   └── ...
└── transforms.json   # Camera poses
```

### Step 2: Run the pipeline

**Basic usage**:
```bash
python src/main.py \
    --video data/video.mp4 \
    --camera-file data/transforms.json \
    --output results/
```

**With images instead of video**:
```bash
python src/main.py \
    --images data/images/ \
    --camera-file data/transforms.json \
    --output results/
```

**High quality mode**:
```bash
python src/main.py \
    --video data/video.mp4 \
    --camera-file data/transforms.json \
    --config configs/high_quality_config.yaml \
    --output results/high_quality/
```

**Fast preview mode**:
```bash
python src/main.py \
    --images data/images/ \
    --camera-file data/transforms.json \
    --config configs/fast_config.yaml \
    --frame-skip 5 \
    --output results/preview/
```

### Step 3: View results

After running the pipeline, check the output directory for:

- `lines_frame_*.jpg` - Sample frames with detected 2D lines
- `line_statistics.png` - Statistics visualization
- `3d_lines.png` - 3D line reconstruction visualization
- `3d_lines.json` - 3D line data
- `gaussians.json` - Initialized Gaussian parameters
- `gaussians_and_lines.ply` - Point cloud visualization

## Using in Your Code

### Basic pipeline usage:
```python
from pipeline import Pipeline3DGS_DeepLSD

# Create config
config = {
    'line_detector': {'type': 'lsd', 'min_length': 15.0},
    'line_reconstructor': {'min_support_views': 2},
    'gaussian_constraint': {'line_weight': 1.0},
    # ... other settings
}

# Run pipeline
pipeline = Pipeline3DGS_DeepLSD(config)
results = pipeline.run_full_pipeline(
    video_path='data/video.mp4',
    camera_file='data/transforms.json',
    output_dir='output/'
)
```

### Using the constraint loss in 3DGS training:
```python
import torch
from gaussian_constraint import GaussianConstraintLoss

# Initialize loss
constraint_loss = GaussianConstraintLoss(
    line_weight=1.0,
    density_weight=0.5,
    alignment_weight=0.3
)

# In training loop
loss, loss_dict = constraint_loss(
    gaussian_positions,
    gaussian_rotations,
    gaussian_scales,
    gaussian_opacities,
    line_starts,
    line_ends
)

# Add to total loss
total_loss = rendering_loss + lambda_line * loss
```

## Camera Pose Format

### transforms.json format (NeRF/3DGS):
```json
{
  "camera_angle_x": 0.857,
  "w": 800,
  "h": 800,
  "frames": [
    {
      "file_path": "./images/frame_00000.jpg",
      "transform_matrix": [
        [0.9, 0.1, 0.0, 1.0],
        [0.0, 0.9, 0.1, 2.0],
        [0.1, 0.0, 0.9, 3.0],
        [0.0, 0.0, 0.0, 1.0]
      ]
    }
  ]
}
```

### COLMAP format:
Place `cameras.txt` and `images.txt` in the same directory, then:
```bash
python src/main.py \
    --images data/images/ \
    --camera-file data/colmap_output/images.txt \
    --output results/
```

## Configuration

Three preset configurations are available:

1. **default_config.yaml** - Balanced quality and speed
2. **high_quality_config.yaml** - Best quality (slower)
3. **fast_config.yaml** - Fast processing (lower quality)

Create your own config by copying and modifying these files.

## Troubleshooting

### "No module named 'numpy'" or similar errors
Install dependencies: `pip install -r requirements.txt`

### "Camera poses file must be provided"
Make sure to specify `--camera-file` pointing to either:
- `transforms.json` file
- COLMAP `images.txt` file (with `cameras.txt` in same directory)

### No lines detected
Try adjusting parameters:
```bash
python src/main.py \
    --min-line-length 10 \
    --max-lines 1000 \
    ...
```

### Low quality reconstruction
Use high quality config:
```bash
python src/main.py \
    --config configs/high_quality_config.yaml \
    ...
```

## Next Steps

1. Check the full README.md for detailed documentation
2. See examples/usage_examples.py for code examples
3. Modify configuration files in configs/ for your use case
4. Integrate the constraint loss into your 3DGS training pipeline

## Support

For issues or questions:
- Open an issue on GitHub
- Check the full documentation in README.md
- Review the example scripts in examples/
