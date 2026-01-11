import os
import cv2
import numpy as np
import torch
from deeplsd.models.deeplsd_inference import DeepLSD

def load_model(ckpt_path):
    # 注意，只允许有这些key
    default_conf = {
        'line_neighborhood': 5,
        'multiscale': False,
        'scale_factors': [1., 1.5],
        'detect_lines': True,
        'line_detection_params': {
            'merge': False,
            'grad_nfa': True,
            'filtering': 'normal',
            'grad_thresh': 3,
        }
    }
    from deeplsd.models.deeplsd_inference import DeepLSD
    model = DeepLSD(default_conf)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model

def run_inference(model, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    # 构造成batch tensor
    tensor = torch.from_numpy(img[None, None, :, :].astype(np.float32) / 255.0)
    with torch.no_grad():
        result = model({'image': tensor})
    # result['lines']: list of ndarrays, 每条线格式[[[r0,c0],[r1,c1]], ...]
    lines = result['lines'][0] if isinstance(result['lines'], list) else result['lines']
    return lines

#自定义可视化配色/线粗，调整 draw_lines()
def draw_lines(img, lines, color=(0,0,255), thickness=2):
    img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim==2 else img.copy()
    h, w = img.shape[:2]
    for line in lines.astype(int):
        # 尝试swap作用
        pt1 = (line[0][0], line[0][1])  # 原来是(col,row)，我们试下(row,col)
        pt2 = (line[1][0], line[1][1])
        # 还可以尝试
        # pt1 = (w-1-line[0][1], line[0][0]) 逆时针旋转
        cv2.line(img_out, pt1, pt2, color, thickness)
    return img_out

if __name__ == "__main__":
    # ========== 1. 配置项 ==========
    # 任选 wireframe or md
    CKPT_PATH = './weights/deeplsd_wireframe.tar'  # 或 './weights/deeplsd_md.tar'
    IMG_DIR = './assets/images'
    OUT_IMG_DIR = './output/output_img'#输出图片目录
    OUT_FEATURE_DIR = './output/output_lines'#输出段特征目录
    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUT_FEATURE_DIR, exist_ok=True)

    model = load_model(CKPT_PATH)
    for fname in sorted(os.listdir(IMG_DIR)):
        if not fname.lower().endswith(('.png','.jpg','.jpeg','.bmp')):
            continue
        fpath = os.path.join(IMG_DIR, fname)
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"无法读取 {fpath}")
            continue
        lines = run_inference(model, img)
        # 2. 画线保存图片
        img_with_lines = draw_lines(img, lines)
        save_img_path = os.path.join(OUT_IMG_DIR, fname)
        cv2.imwrite(save_img_path, img_with_lines)
        # 3. 保存特征（npz，可扩展json/txt）
        basename = os.path.splitext(fname)[0]
        np.savez(os.path.join(OUT_FEATURE_DIR, basename+'_lines.npz'), lines=lines)
        print(f"处理完成: {fname}")

    print("全部图片处理完成！")