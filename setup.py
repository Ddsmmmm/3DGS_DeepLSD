from setuptools import setup, find_packages

setup(
    name="3dgs_deeplsd",
    version="0.1.0",
    description="3D Gaussian Splatting with DeepLSD line constraints for improved reconstruction quality",
    author="3DGS_DeepLSD Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "Pillow>=9.5.0",
        "tqdm>=4.65.0",
        "open3d>=0.17.0",
        "plyfile>=0.9",
        "PyYAML>=6.0",
        "kornia>=0.7.0",
        "einops>=0.6.0",
        "scikit-learn>=1.3.0",
        "scikit-image>=0.21.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
