import os
import sys

# 1. STOP THE OSQP ALGEBRA ERROR (Must be before cvxpy import)
os.environ["CVXPY_SOLVER_MAP_IGNORE"] = "OSQP"

import numpy as np
import cv2
import cvxpy as cp
import json
import pickle
import time
import signal
import psutil
import datetime

# Headless setup for OMV/Linux servers
import matplotlib; matplotlib.use('Agg')

# --- GLOBALS & SIGNALS ---
KEEP_RUNNING = True
SETTINGS_FILE = "settings.json"
CHECKPOINT_FILE = "reconstruction_checkpoint.pkl"

def graceful_exit(signum, frame):
    global KEEP_RUNNING
    print("\n[!] STOP SIGNAL RECEIVED. Finishing current tile and saving...")
    KEEP_RUNNING = False

signal.signal(signal.SIGINT, graceful_exit)

# --- UTILITIES ---
def get_ram_usage():
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except: return 0.0

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            return json.load(f)
    return None

def get_dct_matrix(n):
    d = np.zeros((n, n))
    for k in range(n):
        for i in range(n):
            d[k, i] = np.sqrt(1/n) if k == 0 else np.sqrt(2/n) * np.cos(np.pi * k * (2*i + 1) / (2*n))
    return d

def solve_core(lr_data, config, iters=1000):
    h, w = lr_data.shape
    scale = config.get("SCALE_FACTOR", 2)
    hh, ww = h * scale, w * scale
    lr_norm = lr_data.astype(np.float32) / 255.0
    
    if np.std(lr_norm) < 0.005:
        return cv2.resize(lr_data, (ww, hh), interpolation=cv2.INTER_CUBIC)

    D_h, D_w = get_dct_matrix(hh), get_dct_matrix(ww)
    X_dct = cp.Variable((hh, ww))
    img_rec = D_h.T @ X_dct @ D_w
    
    constraints = [
        cp.abs(img_rec[::scale, ::scale] - lr_norm) <= 0.01,
        img_rec >= 0, img_rec <= 1
    ]
    loss = (cp.norm(X_dct, 1) * config["LAMBDA_MIN"]) + (cp.tv(img_rec) * config["TV_WEIGHT"])
    prob = cp.Problem(cp.Minimize(loss), constraints)
    
    try:
        prob.solve(solver=cp.ECOS, max_iters=iters, abstol=1e-3, reltol=1e-3)
        if prob.status in ["optimal", "optimal_inaccurate"] and X_dct.value is not None:
            return np.clip(img_rec.value * 255.0, 0, 255)
        return cv2.resize(lr_data, (ww, hh), interpolation=cv2.INTER_CUBIC)
    except:
        return cv2.resize(lr_data, (ww, hh), interpolation=cv2.INTER_CUBIC)

def tune_parameters(full_img, scale):
    print("[*] Stage 1: Auto-Tuning for best fidelity...")
    lambdas = [1e-3, 1e-4, 5e-5]
    tv_weights = [0.01, 0.05, 0.1]
    best_psnr = -1
    best_config = {"SCALE_FACTOR": scale, "TILE_SIZE": 64, "OVERLAP": 4}
    
    h, w = full_img.shape[:2]
    patch = full_img[h//2:h//2+64, w//2:w//2+64, 0]
    
    for l in lambdas:
        for tv in tv_weights:
            test_cfg = {"SCALE_FACTOR": scale, "LAMBDA_MIN": l, "TV_WEIGHT": tv}
            recon = solve_core(patch, test_cfg, iters=500)
            ref = cv2.resize(patch, (64*scale, 64*scale), interpolation=cv2.INTER_LANCZOS4)
            mse = np.mean((ref - recon)**2)
            psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 0
            print(f"  > Trial: L={l}, TV={tv} | PSNR: {psnr:.2f}dB")
            if psnr > best_psnr:
                best_psnr = psnr
                best_config.update(test_cfg)
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(best_config, f, indent=4)
    return best_config

def main():
    if len(sys.argv) < 2:
        print("Usage: ./CS_COLOR_V1 <image_path>")
        return

    img_path = sys.argv[1]

    # --- ENHANCED FILE CHECK ---
    if not os.path.exists(img_path):
        print(f"[!] ERROR: '{img_path}' not found in {os.getcwd()}")
        return
    
    full_img = cv2.imread(img_path) 
    if full_img is None:
        print(f"[!] ERROR: OpenCV cannot decode '{img_path}'. Check file permissions or extension.")
        return
    
    H, W, C = full_img.shape
    config = load_settings()
    if config is None:
        config = tune_parameters(full_img, 2)
    else:
        print(f"[*] Settings Loaded: {config}")

    scale = config.get("SCALE_FACTOR", 2)
    ts = config.get("TILE_SIZE", 64)
    stride = ts - config.get("OVERLAP", 4)
    
    output_img = np.zeros((H * scale, W * scale, C), dtype=np.uint8)
    y_coords = range(0, H - ts + 1, stride)
    x_coords = range(0, W - ts + 1, stride)
    all_tiles = [(y, x) for y in y_coords for x in x_coords]
    total_tiles = len(all_tiles)
    start_idx = 0

    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'rb') as f:
            cp_data = pickle.load(f)
            output_img, start_idx = cp_data['img'], cp_data['idx']
            print(f"[*] Resuming from tile {start_idx}")

    # Start Report
    with open("RECON_REPORT.txt", "w") as f:
        f.write(f"RECONSTRUCTION RUN: {datetime.datetime.now()}\n")
        f.write(f"Settings: {json.dumps(config, indent=2)}\n\n")

    print(f"[*] Processing COLOR (RGB): {H}x{W} -> {H*scale}x{W*scale}")
    start_time = time.time()

    for i in range(start_idx, total_tiles):
        if not KEEP_RUNNING:
            with open(CHECKPOINT_FILE, 'wb') as f:
                pickle.dump({'img': output_img, 'idx': i}, f)
            print("[!] Checkpoint saved.")
            sys.exit(0)

        y, x = all_tiles[i]
        t_start = time.time()
        lr_tile = full_img[y:y+ts, x:x+ts]
        
        for c in range(C):
            res = solve_core(lr_tile[:,:,c], config)
            output_img[y*scale:(y+ts)*scale, x*scale:(x+ts)*scale, c] = res.astype(np.uint8)
        
        if i % 25 == 0:
            print(f"[{i+1:05d}/{total_tiles}] Tile({y},{x}) | RAM: {get_ram_usage():.1f}MB")
            with open(CHECKPOINT_FILE, 'wb') as f:
                pickle.dump({'img': output_img, 'idx': i}, f)

    # --- FINAL SAVING ---
    print("[*] Reconstruction finished. Writing image...")
    success = cv2.imwrite("FINAL_COLOR_RECON.png", output_img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    
    if not success:
        print("[!] PNG Save failed. Attempting Emergency JPG...")
        cv2.imwrite("EMERGENCY_RECON.jpg", output_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    total_min = (time.time() - start_time) / 60
    
    # Calculate Accuracy
    h_orig, w_orig = full_img.shape[:2]
    downsampled = cv2.resize(output_img, (w_orig, h_orig), interpolation=cv2.INTER_AREA)
    mse = np.mean((full_img.astype(np.float32) - downsampled.astype(np.float32))**2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100
    
    with open("RECON_REPORT.txt", "a") as f:
        f.write("="*30 + "\n")
        f.write(f"FINAL PSNR ACCURACY: {psnr:.2f} dB\n")
        f.write(f"Total Time: {total_min:.2f} minutes\n")
        f.write("="*30 + "\n")

    if os.path.exists(CHECKPOINT_FILE): os.remove(CHECKPOINT_FILE)
    print(f"[*] Mission Complete. PSNR: {psnr:.2f}dB")

if __name__ == "__main__":
    main()