import os
import sys
import numpy as np
import cv2
import cvxpy as cp
import json
import pickle
import time
import signal
import psutil
import datetime
import csv
import shutil

# 1. STOP THE OSQP ALGEBRA ERROR (Must be before cvxpy import)
os.environ["CVXPY_SOLVER_MAP_IGNORE"] = "OSQP"

# --- GLOBALS & SIGNALS ---
KEEP_RUNNING = True
SETTINGS_FILE = "settings.json"
CHECKPOINT_FILE = "reconstruction_checkpoint.pkl"
TRAIN_CACHE = "training_cache"
MAX_CACHE_GB = 200  
MIN_DRIVE_FREE_GB = 10 

def graceful_exit(signum, frame):
    global KEEP_RUNNING
    print("\n[!] STOP SIGNAL RECEIVED. Cleaning up current tile and exiting...")
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
    return {"SCALE_FACTOR": 2, "TILE_SIZE": 64, "OVERLAP": 4, "LAMBDA_MIN": 1e-4, "TV_WEIGHT": 0.01}

def check_disk_usage(incoming_gb=0):
    usage = shutil.disk_usage(os.getcwd())
    free_gb = usage.free / (1024**3)
    cache_gb = 0
    if os.path.exists(TRAIN_CACHE):
        for dirpath, _, filenames in os.walk(TRAIN_CACHE):
            for f in filenames:
                cache_gb += os.path.getsize(os.path.join(dirpath, f))
    cache_gb /= (1024**3)
    return (free_gb - incoming_gb) > MIN_DRIVE_FREE_GB and (cache_gb + incoming_gb) < MAX_CACHE_GB

# --- PHOTOGRAPHIC NOISE MODEL ---
def add_professional_noise(image, iso_level=1600):
    """Simulates signal-dependent photon shot noise and electronic read noise."""
    img_float = image.astype(np.float32) / 255.0
    shot_intensity = (iso_level / 100) * 0.002
    shot_noise = np.random.normal(0, 1, image.shape) * np.sqrt(img_float + 1e-6) * shot_intensity
    read_noise = np.random.normal(0, 0.005, image.shape)
    noisy_img = img_float + shot_noise + read_noise
    return np.clip(noisy_img * 255.0, 0, 255).astype(np.uint8)

# --- CORE MATH ---
def get_dct_matrix(n):
    d = np.zeros((n, n))
    for k in range(n):
        for i in range(n):
            d[k, i] = np.sqrt(1/n) if k == 0 else np.sqrt(2/n) * np.cos(np.pi * k * (2*i + 1) / (2*n))
    return d

def solve_core(lr_data, config):
    h, w = lr_data.shape
    scale = config.get("SCALE_FACTOR", 2)
    hh, ww = h * scale, w * scale
    lr_norm = lr_data.astype(np.float32) / 255.0
    
    if np.std(lr_norm) < 0.005: # Flat area optimization
        return cv2.resize(lr_data, (ww, hh), interpolation=cv2.INTER_CUBIC)

    D_h, D_w = get_dct_matrix(hh), get_dct_matrix(ww)
    X_dct = cp.Variable((hh, ww))
    img_rec = D_h.T @ X_dct @ D_w
    
    # Adapt tolerance if we are currently training with noise
    tol = 0.02 if config.get("_is_noisy_trial", False) else 0.01
    
    constraints = [
        cp.abs(img_rec[::scale, ::scale] - lr_norm) <= tol,
        img_rec >= 0, img_rec <= 1
    ]
    loss = (cp.norm(X_dct, 1) * config["LAMBDA_MIN"]) + (cp.tv(img_rec) * config["TV_WEIGHT"])
    prob = cp.Problem(cp.Minimize(loss), constraints)
    
    try:
        prob.solve(solver=cp.ECOS, max_iters=500, abstol=1e-3, reltol=1e-3)
        return np.clip(img_rec.value * 255.0, 0, 255) if X_dct.value is not None else cv2.resize(lr_data, (ww, hh))
    except:
        return cv2.resize(lr_data, (ww, hh))

# --- TILING ENGINE ---
def process_full_image(img_in, config, silent=True):
    H, W = img_in.shape[:2]
    C = 1 if len(img_in.shape) == 2 else img_in.shape[2]
    scale, ts = config["SCALE_FACTOR"], config["TILE_SIZE"]
    stride = ts - config["OVERLAP"]
    
    output = np.zeros((H * scale, W * scale, C) if C > 1 else (H * scale, W * scale), dtype=np.uint8)
    
    for y in range(0, H - ts + 1, stride):
        if not KEEP_RUNNING: break
        for x in range(0, W - ts + 1, stride):
            if not KEEP_RUNNING: break
            tile = img_in[y:y+ts, x:x+ts]
            if C > 1:
                for c in range(C):
                    res = solve_core(tile[:,:,c], config)
                    output[y*scale:(y+ts)*scale, x*scale:(x+ts)*scale, c] = res.astype(np.uint8)
            else:
                res = solve_core(tile, config)
                output[y*scale:(y+ts)*scale, x*scale:(x+ts)*scale] = res.astype(np.uint8)
    return output

# --- TRAINING SUITE ---
def run_training(input_dir, use_noise=False):
    print(f"[*] TRAINING INITIATED | ISO Noise: {'ON' if use_noise else 'OFF'}")
    if not os.path.exists(TRAIN_CACHE): os.makedirs(TRAIN_CACHE)
    
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.tif'))]
    lambdas = [1e-4, 5e-5, 1e-5]
    tvs = [0.01, 0.05, 0.1]
    
    results = []
    best_psnr, best_cfg = -1, {}

    try:
        for fname in image_files:
            if not KEEP_RUNNING: break
            orig = cv2.imread(os.path.join(input_dir, fname))
            if orig is None: continue
            
            # Disk space check (roughly 2 copies of downsized image)
            if not check_disk_usage((orig.nbytes / 4) * 2 / (1024**3)):
                print(f"[!] Storage limit reached for {fname}. Skipping.")
                continue

            # Establish training modes for this image
            modes = [("Clean", False)]
            if use_noise: modes.append(("ISO_Noisy", True))

            for mode_name, is_noisy in modes:
                lr = cv2.resize(orig, (orig.shape[1]//2, orig.shape[0]//2), interpolation=cv2.INTER_AREA)
                if is_noisy: lr = add_professional_noise(lr)
                
                tmp_path = os.path.join(TRAIN_CACHE, f"t_{mode_name}_{fname}")
                cv2.imwrite(tmp_path, lr)
                lr_img = cv2.imread(tmp_path)

                print(f"[*] Analyzing {fname} ({mode_name})")

                for l in lambdas:
                    for tv in tvs:
                        if not KEEP_RUNNING: break
                        cfg = {"SCALE_FACTOR": 2, "TILE_SIZE": 64, "OVERLAP": 4, 
                               "LAMBDA_MIN": l, "TV_WEIGHT": tv, "_is_noisy_trial": is_noisy}
                        
                        recon = process_full_image(lr_img, cfg)
                        
                        # PSNR check against the high-res original ground truth
                        ref = cv2.resize(orig, (recon.shape[1], recon.shape[0]))
                        mse = np.mean((ref.astype(np.float32) - recon.astype(np.float32))**2)
                        psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 0
                        
                        results.append([fname, mode_name, l, tv, round(psnr, 4)])
                        if psnr > best_psnr:
                            best_psnr, best_cfg = psnr, cfg
                            print(f"  > NEW BEST: {psnr:.2f}dB (L={l}, TV={tv})")

        if best_cfg:
            best_cfg.pop("_is_noisy_trial", None) # Remove internal flag
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(best_cfg, f, indent=4)
            
            with open("training_log.csv", 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['File', 'Mode', 'Lambda', 'TV', 'PSNR'])
                writer.writerows(results)
            print(f"[*] Training finished. settings.json updated with top performer.")

    finally:
        shutil.rmtree(TRAIN_CACHE, ignore_errors=True)

# --- MAIN ---
def main():
    args = sys.argv[1:]
    if "--train" in args:
        target_dir = args[args.index("--train") + 1]
        use_noise = "--use-noise" in args
        run_training(target_dir, use_noise=use_noise)
        return

    if args:
        img_path = args[0]
        if not os.path.exists(img_path):
            print(f"[!] File not found: {img_path}"); return
            
        config = load_settings()
        img = cv2.imread(img_path)
        print(f"[*] Processing {img_path} with settings: {config}")
        
        start_t = time.time()
        final = process_full_image(img, config)
        
        out_name = "FINAL_RECONSTRUCTION.png"
        cv2.imwrite(out_name, final)
        print(f"[*] Done. Saved to {out_name} in {((time.time()-start_t)/60):.2f}m")
    else:
        print("Commands:\n  Train:   python script.py --train <directory> [--use-noise]\n  Process: python script.py <image_file>")

if __name__ == "__main__":
    main()