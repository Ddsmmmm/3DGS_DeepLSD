"""
Visualization utilities for lines and Gaussians.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional
import open3d as o3d


def draw_lines_on_image(image: np.ndarray,
                        lines: np.ndarray,
                        color: Tuple[int, int, int] = (0, 255, 0),
                        thickness: int = 2) -> np.ndarray:
    """
    Draw line segments on an image.
    
    Args:
        image: Input image (H, W, 3)
        lines: Line segments (N, 4) as [x1, y1, x2, y2]
        color: Line color (B, G, R)
        thickness: Line thickness
        
    Returns:
        Image with drawn lines
    """
    result = image.copy()
    
    for line in lines:
        x1, y1, x2, y2 = line
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        cv2.line(result, pt1, pt2, color, thickness)
    
    return result


def visualize_3d_lines(line_starts: np.ndarray,
                      line_ends: np.ndarray,
                      camera_centers: Optional[np.ndarray] = None,
                      save_path: Optional[str] = None):
    """
    Visualize 3D lines in 3D space using matplotlib.
    
    Args:
        line_starts: Line start points (N, 3)
        line_ends: Line end points (N, 3)
        camera_centers: Optional camera center positions (M, 3)
        save_path: Optional path to save the figure
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw lines
    for start, end in zip(line_starts, line_ends):
        ax.plot([start[0], end[0]],
               [start[1], end[1]],
               [start[2], end[2]],
               'b-', linewidth=2, alpha=0.6)
    
    # Draw camera centers if provided
    if camera_centers is not None:
        ax.scatter(camera_centers[:, 0],
                  camera_centers[:, 1],
                  camera_centers[:, 2],
                  c='r', marker='o', s=50, label='Cameras')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Line Reconstruction')
    ax.legend()
    
    # Equal aspect ratio
    max_range = np.array([
        line_starts.max() - line_starts.min(),
        line_ends.max() - line_ends.min()
    ]).max() / 2.0
    
    mid_x = (line_starts[:, 0].max() + line_starts[:, 0].min()) * 0.5
    mid_y = (line_starts[:, 1].max() + line_starts[:, 1].min()) * 0.5
    mid_z = (line_starts[:, 2].max() + line_starts[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def create_line_point_cloud(line_starts: np.ndarray,
                            line_ends: np.ndarray,
                            points_per_line: int = 100,
                            color: Tuple[float, float, float] = (0, 0, 1)) -> o3d.geometry.PointCloud:
    """
    Create Open3D point cloud from 3D lines.
    
    Args:
        line_starts: Line start points (N, 3)
        line_ends: Line end points (N, 3)
        points_per_line: Number of points to sample per line
        color: RGB color for points (0-1 range)
        
    Returns:
        Open3D point cloud
    """
    points = []
    
    for start, end in zip(line_starts, line_ends):
        t = np.linspace(0, 1, points_per_line)
        line_points = start[np.newaxis, :] + t[:, np.newaxis] * (end - start)[np.newaxis, :]
        points.append(line_points)
    
    points = np.vstack(points)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(color)
    
    return pcd


def visualize_gaussians_and_lines(gaussian_positions: np.ndarray,
                                  gaussian_scales: np.ndarray,
                                  line_starts: np.ndarray,
                                  line_ends: np.ndarray,
                                  save_path: Optional[str] = None):
    """
    Visualize Gaussians and lines together using Open3D.
    
    Args:
        gaussian_positions: Gaussian center positions (N, 3)
        gaussian_scales: Gaussian scales (N, 3)
        line_starts: Line start points (M, 3)
        line_ends: Line end points (M, 3)
        save_path: Optional path to save visualization
    """
    # Create point cloud for Gaussians
    gaussian_pcd = o3d.geometry.PointCloud()
    gaussian_pcd.points = o3d.utility.Vector3dVector(gaussian_positions)
    gaussian_pcd.paint_uniform_color([1, 0, 0])  # Red for Gaussians
    
    # Create point cloud for lines
    line_pcd = create_line_point_cloud(line_starts, line_ends, points_per_line=50)
    
    # Create coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    
    # Visualize
    geometries = [gaussian_pcd, line_pcd, coord_frame]
    
    if save_path:
        # Save as PLY
        o3d.io.write_point_cloud(save_path, gaussian_pcd + line_pcd)
    else:
        o3d.visualization.draw_geometries(geometries,
                                         window_name='Gaussians and Lines',
                                         width=1024,
                                         height=768)


def create_video_with_lines(frames: List[np.ndarray],
                           lines_per_frame: List[np.ndarray],
                           output_path: str,
                           fps: int = 30):
    """
    Create a video showing detected lines on each frame.
    
    Args:
        frames: List of frames (H, W, 3)
        lines_per_frame: List of line arrays, one per frame
        output_path: Path to save output video
        fps: Frames per second
    """
    if len(frames) == 0:
        return
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame, lines in zip(frames, lines_per_frame):
        frame_with_lines = draw_lines_on_image(frame, lines)
        writer.write(frame_with_lines)
    
    writer.release()


def plot_line_statistics(lines_2d_per_frame: List[np.ndarray],
                         lines_3d: np.ndarray,
                         save_path: Optional[str] = None):
    """
    Plot statistics about detected and reconstructed lines.
    
    Args:
        lines_2d_per_frame: List of 2D line arrays per frame
        lines_3d: 3D lines (N, 6) as [x1, y1, z1, x2, y2, z2]
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Number of 2D lines per frame
    num_lines_per_frame = [len(lines) for lines in lines_2d_per_frame]
    axes[0, 0].plot(num_lines_per_frame)
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('Number of 2D Lines')
    axes[0, 0].set_title('2D Line Detections per Frame')
    axes[0, 0].grid(True)
    
    # Distribution of 2D line lengths
    all_2d_lengths = []
    for lines in lines_2d_per_frame:
        if len(lines) > 0:
            lengths = np.sqrt((lines[:, 2] - lines[:, 0])**2 + 
                            (lines[:, 3] - lines[:, 1])**2)
            all_2d_lengths.extend(lengths)
    
    if len(all_2d_lengths) > 0:
        axes[0, 1].hist(all_2d_lengths, bins=50)
        axes[0, 1].set_xlabel('Length (pixels)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('2D Line Length Distribution')
        axes[0, 1].grid(True)
    
    # Distribution of 3D line lengths
    if len(lines_3d) > 0:
        lengths_3d = np.sqrt(
            (lines_3d[:, 3] - lines_3d[:, 0])**2 +
            (lines_3d[:, 4] - lines_3d[:, 1])**2 +
            (lines_3d[:, 5] - lines_3d[:, 2])**2
        )
        axes[1, 0].hist(lengths_3d, bins=30)
        axes[1, 0].set_xlabel('Length (world units)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('3D Line Length Distribution')
        axes[1, 0].grid(True)
    
    # Summary statistics
    summary_text = f"Total 2D lines: {sum(num_lines_per_frame)}\n"
    summary_text += f"Total 3D lines: {len(lines_3d)}\n"
    if len(all_2d_lengths) > 0:
        summary_text += f"Avg 2D length: {np.mean(all_2d_lengths):.2f} px\n"
    if len(lines_3d) > 0:
        summary_text += f"Avg 3D length: {np.mean(lengths_3d):.4f} units"
    
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Summary Statistics')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()
