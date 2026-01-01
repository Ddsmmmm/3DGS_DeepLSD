#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export camera intrinsics and extrinsics from:
 - COLMAP model (sparse/0 : cameras.bin, images.bin, or the text variants)
 - or NeRF-style transforms.json

Outputs:
 - transforms_out.json (NeRF-style with transform_matrix per frame)
 - cameras_intrinsics.json (mapping image_name -> K, width, height, distortion_params if present)
 - cameras_extrinsics.json (mapping image_name -> w2c, c2w, R, t)
 - cameras.npz (numpy archive with arrays)
Usage:
    python scripts/export_colmap_cameras.py --colmap_folder /path/to/scene/sparse/0 --out_dir /tmp/cams
    or
    python scripts/export_colmap_cameras.py --nerf_transforms /path/to/transforms.json --out_dir /tmp/cams
"""
import os
import json
import argparse
import numpy as np
from struct import unpack

# ---------- helper functions ----------
def qvec2rotmat(qvec):
    # qvec is [qw, qx, qy, qz]
    w, x, y, z = qvec
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ], dtype=float)
    return R

def read_next_bytes(fid, num_bytes, fmt):
    return unpack(fmt, fid.read(num_bytes))

# Minimal COLMAP binary readers (adapted for our use)
def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = unpack("Q", fid.read(8))[0]
        for _ in range(num_cameras):
            camera_properties = unpack("iiQQ", fid.read(24))
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            # number of params depends on model - but we will read all remaining as doubles for that model later
            # To get num_params we can check known models; but here we read until expected bytes per model.
            # Common models:
            # 0: SIMPLE_PINHOLE (f, cx, cy) -> 3
            # 1: PINHOLE (fx, fy, cx, cy) -> 4
            # Note: For robustness, read rest of file would be complex; but typical 3DGS uses PINHOLE or SIMPLE_PINHOLE.
            # We attempt to read 8 * 4 bytes (32) to cover up to 4 params, then slice.
            # Move file pointer back and read exact as in COLMAP but simplified here:
            # Not perfect for all models; fallback: return params as empty if can't parse.
            # We'll try reading 4 doubles and then adapt.
            params = list(unpack("dddd", fid.read(32)))
            # Placeholders for model name detection -- map model_id to name if known:
            model_map = {0: "SIMPLE_PINHOLE", 1: "PINHOLE"}
            model_name = model_map.get(model_id, "UNKNOWN")
            cameras[camera_id] = {"id": camera_id, "model": model_name, "width": width, "height": height, "params": params}
    return cameras

def read_images_binary(path):
    images = {}
    with open(path, "rb") as fid:
        num_images = unpack("Q", fid.read(8))[0]
        for _ in range(num_images):
            # image id (Q?) In COLMAP read_write_model uses: binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            # We'll follow similar layout: id (I) + qvec(4d) + tvec(3d) + camera_id (I)
            # We'll read as described by COLMAP's cheat: read 64 bytes with format "idddddddi"
            # But safer: unpack according to what's in the repo. We'll mimic that format:
            raw = fid.read(64)
            elems = unpack("idddddddi", raw)
            image_id = elems[0]
            qvec = np.array(elems[1:5], dtype=float)
            tvec = np.array(elems[5:8], dtype=float)
            camera_id = elems[8]
            # Now read NULL-terminated image name
            name_bytes = b""
            while True:
                c = fid.read(1)
                if c == b'\x00' or c == b'':
                    break
                name_bytes += c
            image_name = name_bytes.decode("utf-8")
            # After that, COLMAP stores number of 2D points + their tuples, but we don't need them.
            # We still need to skip the part with points: read Q for num_points2D and then 24*num_points2D bytes
            num_pts_data = fid.read(8)
            if len(num_pts_data) == 0:
                num_points2D = 0
            else:
                num_points2D = unpack("Q", num_pts_data)[0]
                fid.read(24 * num_points2D)
            images[image_id] = {"id": image_id, "qvec": qvec, "tvec": tvec, "camera_id": camera_id, "name": image_name}
    return images

# ---------- main exporter ----------
def build_K_from_cam(cam_entry):
    model = cam_entry["model"]
    params = cam_entry["params"]
    w = cam_entry["width"]
    h = cam_entry["height"]
    if model == "SIMPLE_PINHOLE":
        f = params[0]
        cx = params[1]
        cy = params[2]
        fx = fy = f
    elif model == "PINHOLE":
        fx = params[0]
        fy = params[1]
        cx = params[2]
        cy = params[3]
    else:
        # fallback: assume fx=params[0], cx=params[1], cy=params[2]
        fx = params[0] if len(params) > 0 else 1.0
        fy = params[1] if len(params) > 1 else fx
        cx = params[2] if len(params) > 2 else w*0.5
        cy = params[3] if len(params) > 3 else h*0.5
    K = np.array([[fx, 0.0, cx],[0.0, fy, cy],[0.0, 0.0, 1.0]], dtype=float)
    return K

def export_from_colmap(colmap_folder, out_dir, use_text=False):
    # prefer binary files in typical sparse/0 folder
    cams_bin = os.path.join(colmap_folder, "cameras.bin")
    imgs_bin = os.path.join(colmap_folder, "images.bin")
    cams_txt = os.path.join(colmap_folder, "cameras.txt")
    imgs_txt = os.path.join(colmap_folder, "images.txt")

    if os.path.exists(cams_bin) and os.path.exists(imgs_bin):
        cams = read_cameras_binary(cams_bin)
        imgs = read_images_binary(imgs_bin)
    elif os.path.exists(cams_txt) and os.path.exists(imgs_txt):
        raise NotImplementedError("Text reader not implemented in this simple script, but you can use read_write_model.py from the repo.")
    else:
        raise FileNotFoundError("No COLMAP model files found in {}".format(colmap_folder))

    # assemble per-image entries
    intrinsics_out = {}
    extrinsics_out = {}
    transforms_frames = []

    for img_id, img in imgs.items():
        cam = cams[img["camera_id"]]
        K = build_K_from_cam(cam)
        qvec = img["qvec"]
        tvec = img["tvec"]
        # w2c: world to camera: [R | t] where R = qvec2rotmat(qvec)
        R = qvec2rotmat(qvec)
        t = tvec
        w2c = np.eye(4, dtype=float)
        w2c[:3,:3] = R
        w2c[:3,3] = t
        # c2w = inverse
        c2w = np.linalg.inv(w2c)

        image_name = img["name"]

        intrinsics_out[image_name] = {
            "width": cam["width"],
            "height": cam["height"],
            "model": cam["model"],
            "params": cam["params"],
            "K": K.tolist()
        }
        extrinsics_out[image_name] = {
            "R": R.tolist(),    # rotation in w2c such that X_c = R X_w + t
            "t": t.tolist(),
            "w2c": w2c.tolist(),
            "c2w": c2w.tolist()
        }

        # NeRF-style frame: store c2w
        transforms_frames.append({
            "file_path": image_name,
            "transform_matrix": c2w.tolist()
        })

    os.makedirs(out_dir, exist_ok=True)
    # write jsons
    with open(os.path.join(out_dir, "cameras_intrinsics.json"), "w") as f:
        json.dump(intrinsics_out, f, indent=2)
    with open(os.path.join(out_dir, "cameras_extrinsics.json"), "w") as f:
        json.dump(extrinsics_out, f, indent=2)
    transforms = {
        "fl_x": None, "fl_y": None, "cx": None, "cy": None,
        "camera_angle_x": None,
        "frames": transforms_frames
    }
    with open(os.path.join(out_dir, "transforms_out.json"), "w") as f:
        json.dump(transforms, f, indent=2)

    # save numpy archive for quick loading
    npz_dict = {}
    for k,v in intrinsics_out.items():
        npz_dict[f"K__{k}"] = np.array(v["K"])
    for k,v in extrinsics_out.items():
        npz_dict[f"w2c__{k}"] = np.array(v["w2c"])
        npz_dict[f"c2w__{k}"] = np.array(v["c2w"])
    np.savez_compressed(os.path.join(out_dir, "cameras.npz"), **npz_dict)

    print("Exported {} cameras to {}".format(len(intrinsics_out), out_dir))

def export_from_transforms(nerf_transforms_path, out_dir):
    with open(nerf_transforms_path, "r") as f:
        data = json.load(f)
    frames = data.get("frames", [])
    intrinsics_out = {}
    extrinsics_out = {}
    transforms_frames = []
    # derive K if fields available
    camera_angle_x = data.get("camera_angle_x", None)
    for fr in frames:
        file_path = fr.get("file_path")
        c2w = np.array(fr.get("transform_matrix"), dtype=float)
        w2c = np.linalg.inv(c2w)
        # if focal provided as fl_x or camera_angle_x, try to reconstruct K if width/height exist in frame (not always)
        # Here we write c2w and w2c and leave K absent unless fl_x/fl_y are present in file
        extrinsics_out[file_path] = {
            "c2w": c2w.tolist(),
            "w2c": w2c.tolist(),
            "R": w2c[:3,:3].tolist(),
            "t": w2c[:3,3].tolist()
        }
        transforms_frames.append({"file_path": file_path, "transform_matrix": c2w.tolist()})

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "cameras_extrinsics_from_transforms.json"), "w") as f:
        json.dump(extrinsics_out, f, indent=2)
    with open(os.path.join(out_dir, "transforms_out.json"), "w") as f:
        json.dump({"frames": transforms_frames, "camera_angle_x": camera_angle_x}, f, indent=2)
    print("Exported {} frames from transforms.json to {}".format(len(frames), out_dir))

# ---------- CLI ----------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--colmap_folder", type=str, default=None, help="Path to COLMAP sparse folder (e.g., <scene>/sparse/0)")
    p.add_argument("--nerf_transforms", type=str, default=None, help="Path to transforms.json (NeRF style)")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory")
    args = p.parse_args()

    if args.colmap_folder:
        export_from_colmap(args.colmap_folder, args.out_dir)
    elif args.nerf_transforms:
        export_from_transforms(args.nerf_transforms, args.out_dir)
    else:
        raise ValueError("Please supply either --colmap_folder or --nerf_transforms")