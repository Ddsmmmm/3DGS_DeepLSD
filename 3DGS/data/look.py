#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lines_2d_to_3d.py

Usage examples:
  # depth mode (recommended)
  python scripts/lines_2d_to_3d.py \
    --intrinsics out/cameras_intrinsics.json \
    --extrinsics out/cameras_extrinsics.json \
    --lines lines_2d.json \
    --depth_dir depths/ \
    --out_dir out_lines \
    --mode depth \
    --samples_per_line 30

  # multi-view mode (need matches mapping)
  python scripts/lines_2d_to_3d.py \
    --intrinsics out/cameras_intrinsics.json \
    --extrinsics out/cameras_extrinsics.json \
    --matches matches_lines_groups.json \
    --out_dir out_lines \
    --mode multiview

Inputs:
 - intrinsics: JSON produced by export_colmap_cameras.py (cameras_intrinsics.json)
 - extrinsics: JSON produced by export_colmap_cameras.py (cameras_extrinsics.json)
 - lines OR matches:
     * lines: a single JSON mapping image_name -> list of line dicts:
         { "image.jpg": [ {"x1":..., "y1":..., "x2":..., "y2":..., "score":...}, ... ], ... }
       or per-image JSON files in a directory (image.jpg.json)
     * matches (for multiview mode): mapping line_group_id -> list of observations:
         { "L0001": [ {"image":"imgA.jpg","x1":..,"y1":..,"x2":..,"y2":..}, ... ], ... }
 - depth_dir (optional, for depth mode): per-image depth maps named like image_name + ext (default .png).
   Depth format handling: supports 16-bit PNG where values are linear depth in meters (or inverse if --depth_inverse).
   Use --depth_scale to scale read values to meters if needed.

Outputs:
 - out_dir/lines_3d.json
 - out_dir/lines_3d_sampled.ply
"""
import os
import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image

# ------------------ helpers -------------------
def load_json_maybe_file_or_dir(path):
    path = Path(path)
    if path.is_file():
        with open(path, "r") as f:
            return json.load(f)
    elif path.is_dir():
        data = {}
        for p in path.glob("*.json"):
            name = p.stem
            with open(p, "r") as f:
                data[name] = json.load(f)
        return data
    else:
        raise FileNotFoundError(path)

def read_intrinsics(path):
    with open(path, "r") as f:
        return json.load(f)

def read_extrinsics(path):
    with open(path, "r") as f:
        return json.load(f)

def inv_K_from_K(K):
    return np.linalg.inv(np.array(K))

def qvec2rotmat(qvec):
    w, x, y, z = qvec
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ], dtype=float)
    return R

def line_from_points_uv(p1, p2):
    # p1, p2 are (u,v) image pixel coords
    x1 = np.array([p1[0], p1[1], 1.0])
    x2 = np.array([p2[0], p2[1], 1.0])
    l = np.cross(x1, x2)
    return l / (np.linalg.norm(l[:2]) + 1e-12)

def plane_from_image_line(line_uv, K, R, t):
    # line_uv: [x1,y1,x2,y2] or l homogeneous
    if len(line_uv) == 4:
        l = line_from_points_uv((line_uv[0], line_uv[1]), (line_uv[2], line_uv[3]))
    else:
        l = np.array(line_uv)
    K = np.array(K)
    n_c = K.T.dot(l)        # plane normal in camera coords
    n_w = R.T.dot(n_c)      # plane normal in world coords
    d = float(n_c.dot(t))   # plane: n_w^T X_w + d = 0
    return n_w, d

def intersect_two_planes(n1, d1, n2, d2):
    v = np.cross(n1, n2)
    nv = np.linalg.norm(v)
    if nv < 1e-8:
        return None
    v = v / nv
    A = np.vstack((n1, n2, v))
    b = -np.array([d1, d2, 0.0])
    try:
        X0 = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return X0, v

def fit_line_PCA(points):
    P = np.array(points)
    if P.shape[0] < 2:
        return None
    centroid = P.mean(axis=0)
    U, S, Vt = np.linalg.svd((P - centroid).T)
    direction = Vt[0]
    # length estimated by projecting points onto direction
    coords = (P - centroid) @ direction
    length = float(coords.max() - coords.min())
    return {"point": centroid.tolist(), "direction": direction.tolist(), "length": length, "n_points": int(P.shape[0])}

def load_depth_map(depth_path, depth_inverse=False, depth_scale=1.0):
    img = Image.open(depth_path)
    arr = np.array(img)
    # typical 16-bit PNG
    if arr.dtype == np.uint16 or arr.dtype == np.uint32:
        depth = arr.astype(np.float32) * depth_scale
    else:
        depth = arr.astype(np.float32) * depth_scale
    if depth_inverse:
        # avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            depth = np.where(depth > 0, 1.0 / depth, 0.0)
    return depth

def save_lines_json(lines, out_path):
    with open(out_path, "w") as f:
        json.dump(lines, f, indent=2)

def save_ply_points(points, out_path):
    # points: N x 3 or N x 6 (with color)
    P = np.array(points)
    has_color = P.shape[1] == 6
    n = P.shape[0]
    with open(out_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_color:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            if has_color:
                x,y,z,r,g,b = P[i]
                f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")
            else:
                x,y,z = P[i]
                f.write(f"{x} {y} {z}\n")

# ------------------ main pipeline -------------------
def run_depth_mode(intrinsics_path, extrinsics_path, lines_input, depth_dir, out_dir, samples_per_line=30,
                   depth_ext=".png", depth_inverse=False, depth_scale=1.0, min_valid=5):
    intr = read_intrinsics(intrinsics_path)
    extr = read_extrinsics(extrinsics_path)
    # lines_input can be a JSON file mapping image->list or a dir of per-image JSONs
    if os.path.isdir(lines_input):
        # gather per-image json files
        lines_map = {}
        for p in Path(lines_input).glob("*.json"):
            imgname = p.stem
            with open(p, "r") as f:
                lines_map[imgname] = json.load(f)
    else:
        with open(lines_input, "r") as f:
            lines_map = json.load(f)

    os.makedirs(out_dir, exist_ok=True)
    lines_3d = []
    sampled_points = []

    for img_name, lines in lines_map.items():
        # find matching intr/extr entry keys: exported intrinsics may use full image filename
        # try exact match, then try removing path, etc.
        key = img_name
        if key not in intr:
            # try matching basename
            key = os.path.basename(img_name)
        if key not in intr or key not in extr:
            print(f"[WARN] camera metadata for {img_name} not found in intrinsics/extrinsics.")
            continue

        K = np.array(intr[key]["K"])
        Kinv = np.linalg.inv(K)
        cam_ex = extr[key]
        c2w = np.array(cam_ex["c2w"])
        # load depth
        depth_path = None
        cand = Path(depth_dir) / (key + depth_ext)
        if not cand.exists():
            cand = Path(depth_dir) / (os.path.splitext(key)[0] + depth_ext)
        if not cand.exists():
            print(f"[WARN] depth for {key} not found at {cand}. skipping image.")
            continue
        depth = load_depth_map(str(cand), depth_inverse=depth_inverse, depth_scale=depth_scale)

        for li, L in enumerate(lines):
            # expect L has keys x1,y1,x2,y2
            x1,y1,x2,y2 = float(L["x1"]), float(L["y1"]), float(L["x2"]), float(L["y2"])
            pts2d = np.stack([np.linspace(x1,x2,samples_per_line), np.linspace(y1,y2,samples_per_line)], axis=-1)
            pts3d_world = []
            for (u,v) in pts2d:
                ui = int(round(u)); vi = int(round(v))
                if vi < 0 or vi >= depth.shape[0] or ui < 0 or ui >= depth.shape[1]:
                    continue
                z = float(depth[vi, ui])
                if z <= 0:
                    continue
                x_c = Kinv.dot(np.array([u*z, v*z, z]))
                # transform to world: X_w = c2w @ [Xc;1]
                Xc_h = np.array([x_c[0], x_c[1], x_c[2], 1.0])
                Xw_h = c2w.dot(Xc_h)
                pts3d_world.append(Xw_h[:3])
            if len(pts3d_world) >= min_valid:
                fit = fit_line_PCA(pts3d_world)
                if fit is not None:
                    fit["id"] = f"{key}__{li}"
                    fit["source_images"] = [key]
                    lines_3d.append(fit)
                    # sample points for PLY visualization
                    cen = np.array(fit["point"])
                    dirv = np.array(fit["direction"])
                    Llen = fit["length"]
                    # if length == 0, fallback to spread
                    if Llen <= 1e-6:
                        Llen = np.linalg.norm(np.max(pts3d_world, axis=0) - np.min(pts3d_world, axis=0))
                    n_samples = max(2, int(min(50, Llen)))  # heuristic
                    tvals = np.linspace(-0.5, 0.5, max(8, int(samples_per_line/2)))
                    for t in tvals:
                        p = cen + dirv * (t * Llen)
                        sampled_points.append([p[0], p[1], p[2], 255, 0, 0])
            else:
                # not enough valid samples
                continue

    # save
    save_lines_json(lines_3d, os.path.join(out_dir, "lines_3d.json"))
    save_ply_points(sampled_points, os.path.join(out_dir, "lines_3d_sampled.ply"))
    print(f"Depth-mode: exported {len(lines_3d)} 3D lines to {out_dir}")

def run_multiview_mode(intrinsics_path, extrinsics_path, matches_json, out_dir):
    intr = read_intrinsics(intrinsics_path)
    extr = read_extrinsics(extrinsics_path)
    with open(matches_json, "r") as f:
        matches = json.load(f)

    os.makedirs(out_dir, exist_ok=True)
    lines_3d = []
    sampled_points = []

    for gid, observations in matches.items():
        planes = []
        pts_candidates = []
        for obs in observations:
            image = obs["image"]
            key = image if image in intr else os.path.basename(image)
            if key not in intr or key not in extr:
                print(f"[WARN] image {image} not found in intr/extr. skipping obs.")
                continue
            K = np.array(intr[key]["K"])
            cam_ex = extr[key]
            # extract R,t from w2c convention in original exporter: w2c: Xc = R Xw + t
            # the saved extrinsics contain R and t in that convention.
            R = np.array(cam_ex["R"])
            t = np.array(cam_ex["t"])
            line_uv = [obs["x1"], obs["y1"], obs["x2"], obs["y2"]]
            n_w, d = plane_from_image_line(line_uv, K, R, t)
            planes.append((n_w, d))

        # compute pairwise intersections to get candidate 3D points
        for i in range(len(planes)):
            for j in range(i+1, len(planes)):
                res = intersect_two_planes(planes[i][0], planes[i][1], planes[j][0], planes[j][1])
                if res is not None:
                    X0, v = res
                    pts_candidates.append(X0)
        if len(pts_candidates) < 2:
            # can't fit
            continue
        fit = fit_line_PCA(pts_candidates)
        if fit is None:
            continue
        fit["id"] = gid
        fit["source_images"] = [obs["image"] for obs in observations]
        lines_3d.append(fit)
        # sampled points
        cen = np.array(fit["point"])
        dirv = np.array(fit["direction"])
        Llen = fit["length"] if fit["length"] > 0 else 1.0
        tvals = np.linspace(-0.5, 0.5, 40)
        for t in tvals:
            p = cen + dirv * (t * Llen)
            sampled_points.append([p[0], p[1], p[2], 0, 255, 0])

    save_lines_json(lines_3d, os.path.join(out_dir, "lines_3d.json"))
    save_ply_points(sampled_points, os.path.join(out_dir, "lines_3d_sampled.ply"))
    print(f"Multi-view mode: exported {len(lines_3d)} 3D lines to {out_dir}")

# ------------------ CLI -------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--intrinsics", required=True, help="path to cameras_intrinsics.json")
    p.add_argument("--extrinsics", required=True, help="path to cameras_extrinsics.json")
    p.add_argument("--lines", default=None, help="path to lines JSON mapping image->list OR directory with per-image JSONs")
    p.add_argument("--matches", default=None, help="path to matches JSON for multiview mode")
    p.add_argument("--depth_dir", default=None, help="directory with depth maps (for depth mode)")
    p.add_argument("--out_dir", required=True, help="output directory")
    p.add_argument("--mode", choices=["depth","multiview"], required=True)
    p.add_argument("--samples_per_line", type=int, default=30)
    p.add_argument("--depth_ext", default=".png")
    p.add_argument("--depth_inverse", action="store_true", help="if depth maps are inverse-depth")
    p.add_argument("--depth_scale", type=float, default=1.0, help="scale to convert depth values to meters")
    args = p.parse_args()

    if args.mode == "depth":
        assert args.lines is not None and args.depth_dir is not None, "depth mode needs --lines and --depth_dir"
        run_depth_mode(args.intrinsics, args.extrinsics, args.lines, args.depth_dir,
                       args.out_dir, samples_per_line=args.samples_per_line,
                       depth_ext=args.depth_ext, depth_inverse=args.depth_inverse, depth_scale=args.depth_scale)
    else:
        assert args.matches is not None, "multiview mode needs --matches"
        run_multiview_mode(args.intrinsics, args.extrinsics, args.matches, args.out_dir)