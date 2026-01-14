"""
3DGS_DeepLSD: 3D Gaussian Splatting with DeepLSD Line Constraints

A 3D reconstruction method that improves 3DGS model quality by constraining
Gaussian ellipsoids using 3D lines reconstructed from 2D line detections.
"""

__version__ = "0.1.0"
__author__ = "3DGS_DeepLSD Contributors"

from .pipeline import Pipeline3DGS_DeepLSD

__all__ = ['Pipeline3DGS_DeepLSD']
