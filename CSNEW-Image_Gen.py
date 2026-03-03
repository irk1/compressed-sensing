import numpy as np
import cv2
import cvxpy as cp
import json
import os
import datetime
import sys
import pickle
from skimage.metrics import peak_signal_noise_ratio as psnr

# Headless setup: No GUI required
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
SETTINGS_FILE = "settings.json"
REPORT_FILE = "80MP_run_report.txt"
CHECKPOINT_FILE = "reconstruction_checkpoint.pkl"

DEFAULT_SETTINGS = {
    "SCALE_FACTOR": 2, 
    "TUNING_TRIALS": 10,
    "STALL_COUNT": 0,
    "LAMBDA_MIN": 1e-4, "LAMBDA_MAX": 1e-2,
    "TV_WEIGHT": 0.05, "EDGE_WEIGHT": 0.1,
    "BEST_PSNR_FOUND": 0.0,
    "TILE_SIZE": 64,  # Size of Low-Res tile
    "OVERLAP": 4      # Pixels for seamless blending
}

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            return {**DEFAULT_SETTINGS, **json.load(f)}
    return DEFAULT_SETTINGS

def save_settings(config):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(config, f, indent=4)

def get_dct_matrix(n):
    d = np.zeros((n, n))
    for k in range(n):
        for i in range(n):
            d[k, i] = np.sqrt(1/n) if k == 0 else np.sqrt(2/n) * np.cos(np.pi * k * (2*i + 1) / (2*n))
    return d

def solve_core(lr_data, config, iters=1500):
    """The heavy lifting solver for a single tile."""
    h, w = lr_data.shape
    scale = config["SCALE_FACTOR"]
    hh, ww = h * scale, w * scale
    D_h, D_w = get_dct_matrix(hh), get_dct_matrix(ww)
    
    X_dct = cp.Variable((hh, ww))
    img_rec = D_h.T @ X_dct @ D_w
    
    constraints = [img_rec[::scale, ::scale] == lr_data]
    loss = (cp.norm(X_dct, 1) * config["LAMBDA_MIN"]) + \
           (cp.tv(img_rec) * config["TV_WEIGHT"]) + \
           (cp.norm(cp.diff(img_rec, axis=0), 2) * config["EDGE_WEIGHT"]) + \
           (cp.norm(cp.diff(img_rec, axis=1), 2) * config["EDGE_WEIGHT"])
    
    prob = cp.Problem(cp.Minimize(loss), constraints)
    try:
        prob.solve(solver=cp.ECOS, max_iters=iters)
        return np.clip(img_rec.value, 0, 255) if X_dct.value is not None else None
    except:
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: ./reconstructor <image_path>")
        return

    config = load_settings()
    img_path = sys.argv[1]
    full_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if full_img is None: return
    
    H, W = full_img.shape
    scale = config["SCALE_FACTOR"]
    
    # --- PHASE 1: HIERARCHICAL TUNING ---
    print(f"[{datetime.datetime.now()}] Stage 1: Tuning on random patches...")
    stages = [32, 64]
    for s in stages:
        y, x = np.random.randint(0, H-s), np.random.randint(0, W-s)
        gt_patch = full_img[y:y+s, x:x+s].astype(np.float32)
        lr_patch = gt_patch[::scale, ::scale]
        # In a real run, you'd loop trials here to update LAMBDA_MIN/MAX
        solve_core(lr_patch, config) 
    
    save_settings(config)

    # --- PHASE 2: TILED RECONSTRUCTION WITH CHECKPOINTS ---
    output_img = np.zeros((H * scale, W * scale), dtype=np.uint8)
    start_y, start_x = 0, 0
    
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'rb') as f:
            checkpoint = pickle.load(f)
            output_img = checkpoint['img']
            start_y, start_x = checkpoint['y'], checkpoint['x']
            print(f"Resuming from checkpoint at Tile Y:{start_y} X:{start_x}")

    ts = config["TILE_SIZE"]
    stride = ts - config["OVERLAP"]
    
    print(f"[{datetime.datetime.now()}] Stage 2: Reconstructing 80MP image...")
    
    count = 0
    for y in range(start_y, H - ts, stride):
        for x in range(0, W - ts, stride):
            # Skip tiles already processed if resuming
            if y == start_y and x < start_x: continue
            
            lr_tile = full_img[y:y+ts, x:x+ts].astype(np.float32)
            hr_tile = solve_core(lr_tile, config)
            
            if hr_tile is not None:
                ys, xs = y * scale, x * scale
                ye, xe = ys + (ts * scale), xs + (ts * scale)
                output_img[ys:ye, xs:xe] = hr_tile.astype(np.uint8)
            
            count += 1
            if count % 50 == 0:
                print(f"Progress: Tile {count} processed. Saving checkpoint...")
                with open(CHECKPOINT_FILE, 'wb') as f:
                    pickle.dump({'img': output_img, 'y': y, 'x': x}, f)

    # --- PHASE 3: FINAL REPORT ---
    cv2.imwrite("FINAL_80MP_RESULT.png", output_img)
    with open(REPORT_FILE, "w") as f:
        f.write(f"Finish Time: {datetime.datetime.now()}\n")
        f.write(f"Final Settings: {json.dumps(config, indent=2)}")
    
    if os.path.exists(CHECKPOINT_FILE): os.remove(CHECKPOINT_FILE)
    print("Reconstruction Complete.")

if __name__ == "__main__":
    main()