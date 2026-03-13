import os
import sys
import numpy as np
import cv2
import cvxpy as cp
import json
import time
import signal
import psutil
import rawpy
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# 1. HARDWARE & SIGNAL OPTIMIZATION
os.environ["CVXPY_SOLVER_MAP_IGNORE"] = "OSQP"
KEEP_RUNNING = True
SETTINGS_FILE = "settings.json"

def graceful_exit(signum, frame):
    global KEEP_RUNNING
    print("\n[!] STOP SIGNAL RECEIVED. Finishing current high-precision tiles...")
    KEEP_RUNNING = False

signal.signal(signal.SIGINT, graceful_exit)

# --- SETTINGS LOADER ---
def load_settings():
    defaults = {
        "SCALE_FACTOR": 2, 
        "TILE_SIZE": 64, 
        "OVERLAP": 12, 
        "LAMBDA_MIN": 1e-5, 
        "TV_WEIGHT": 0.002, 
        "MAX_THREADS": 40, 
        "MAX_RAM_GB": 30
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                user_cfg = json.load(f)
                return {**defaults, **user_cfg}
        except:
            return defaults
    return defaults

# --- HIGH-BIT DEPTH LOADING (FLOAT32 COMPATIBILITY) ---
def load_image(path):
    try:
        if path.lower().endswith(('.dng', '.nef', '.cr2', '.arw', '.orf')):
            with rawpy.imread(path) as raw:
                try:
                    rgb = raw.postprocess(
                        use_camera_wb=True, 
                        no_auto_bright=False, 
                        bright=1.0, 
                        output_bps=16, 
                        demosaic_algorithm=rawpy.DemosaicAlgorithm.DHT
                    )
                except:
                    rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False, output_bps=16)
                
                bgr_16 = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                return bgr_16.astype(np.float32) / 65535.0
        else:
            bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if bgr is None: return None
            return bgr.astype(np.float32) / (255.0 if bgr.dtype == np.uint8 else 65535.0)
    except Exception as e:
        print(f"[ERROR] Loading Failed: {e}"); return None

# --- THE "HEAVY" SOLVER (5,000 ITERATIONS) ---
def get_dct_matrix(n):
    d = np.zeros((n, n))
    for k in range(n):
        for i in range(n):
            d[k, i] = np.sqrt(1/n) if k == 0 else np.sqrt(2/n) * np.cos(np.pi * k * (2*i + 1) / (2*n))
    return d

def solve_tile(args):
    lr_tile, scale, l_min, tv_w = args
    h, w = lr_tile.shape
    hh, ww = h * scale, w * scale
    D_h, D_w = get_dct_matrix(hh), get_dct_matrix(ww)
    X_dct = cp.Variable((hh, ww))
    img_rec = D_h.T @ X_dct @ D_w
    
    constraints = [
        cp.abs(img_rec[::scale, ::scale] - lr_tile) <= 0.001,
        img_rec >= 0, img_rec <= 1
    ]
    loss = (cp.norm(X_dct, 1) * l_min) + (cp.tv(img_rec) * tv_w)
    prob = cp.Problem(cp.Minimize(loss), constraints)
    
    try:
        prob.solve(solver=cp.ECOS, max_iters=5000, abstol=1e-8)
        if X_dct.value is not None:
            return np.clip(img_rec.value, 0, 1).astype(np.float32)
        return cv2.resize(lr_tile, (ww, hh), interpolation=cv2.INTER_LANCZOS4)
    except:
        return cv2.resize(lr_tile, (ww, hh), interpolation=cv2.INTER_LANCZOS4)

# --- CORE ENGINE ---
def process_full(img_float, config, l_override=None, t_override=None):
    l_min = l_override if l_override is not None else config["LAMBDA_MIN"]
    tv_w = t_override if t_override is not None else config["TV_WEIGHT"]
    
    lab_32f = cv2.cvtColor(img_float, cv2.COLOR_BGR2Lab)
    channels = cv2.split(lab_32f)
    up_chans = []
    max_ram_bytes = config["MAX_RAM_GB"] * (1024**3)

    for chan in channels:
        H, W = chan.shape
        scale, ts = config["SCALE_FACTOR"], config["TILE_SIZE"]
        stride = ts - config["OVERLAP"]
        
        work, pos = [], []
        for y in range(0, H - ts + 1, stride):
            for x in range(0, W - ts + 1, stride):
                work.append((chan[y:y+ts, x:x+ts], scale, l_min, tv_w))
                pos.append((y*scale, x*scale))

        out = np.zeros((H * scale, W * scale), dtype=np.float32)
        with ProcessPoolExecutor(max_workers=config["MAX_THREADS"]) as ex:
            futures = []
            for item in work:
                while psutil.virtual_memory().available < (psutil.virtual_memory().total - max_ram_bytes):
                    time.sleep(0.5)
                if not KEEP_RUNNING: break
                futures.append(ex.submit(solve_tile, item))

            for i, f in enumerate(futures):
                y, x = pos[i]
                out[y:y+(ts*scale), x:x+(ts*scale)] = f.result()
        up_chans.append(out)

    merged = cv2.merge(up_chans)
    bgr_32f = cv2.cvtColor(merged, cv2.COLOR_Lab2BGR)
    bgr_8 = (np.clip(bgr_32f, 0, 1) * 255).astype(np.uint8)
    cleaned_8 = cv2.fastNlMeansDenoisingColored(bgr_8, None, 3, 3, 7, 21)
    return (cleaned_8.astype(np.uint16) * 257)

# --- RESTORED FOLDER TESTING LOGIC ---
def run_testing_mode(img_float, config, base_path):
    print(f"[!] TEST GRID: Generating 9 variations for {os.path.basename(base_path)}...")
    lambdas = [1e-6, 1e-5, 1e-4]
    tvs = [0.001, 0.002, 0.005]
    for l_val in lambdas:
        for t_val in tvs:
            if not KEEP_RUNNING: return
            print(f"  [>] Solving: Lambda={l_val}, TV={t_val}")
            res = process_full(img_float, config, l_val, t_val)
            cv2.imwrite(f"{base_path}_L{l_val}_T{t_val}.tif", res, [cv2.IMWRITE_TIFF_COMPRESSION, 32946])

# --- MAIN ---
def main():
    args = sys.argv[1:]
    config = load_settings()
    valid_exts = ('.dng', '.nef', '.cr2', '.arw', '.jpg', '.png', '.tif')

    if "--test" in args:
        path = args[args.index("--test") + 1]
        if os.path.isdir(path):
            files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(valid_exts)])
            print(f"[*] FOLDER TRAINING: Processing {len(files)} files...")
            for f in files:
                if not KEEP_RUNNING: break
                img = load_image(f)
                if img is not None: run_testing_mode(img, config, os.path.splitext(f)[0])
        else:
            img = load_image(path)
            if img is not None: run_testing_mode(img, config, os.path.splitext(path)[0])
            
    elif "--batch" in args:
        batch_dir = args[args.index("--batch") + 1]
        files = sorted([os.path.join(batch_dir, f) for f in os.listdir(batch_dir) if f.lower().endswith(valid_exts) and "_upscaled" not in f])
        for f in files:
            if not KEEP_RUNNING: break
            print(f"[*] BATCH RUN: {os.path.basename(f)}")
            img = load_image(f)
            if img is not None:
                res = process_full(img, config)
                cv2.imwrite(f"{os.path.splitext(f)[0]}_upscaled.tif", res, [cv2.IMWRITE_TIFF_COMPRESSION, 32946])
    elif args:
        img = load_image(args[0])
        if img is not None:
            res = process_full(img, config)
            cv2.imwrite(f"{os.path.splitext(args[0])[0]}_upscaled.tif", res, [cv2.IMWRITE_TIFF_COMPRESSION, 32946])
    else:
        print("Usage: ./CSNEW-Image_Gen --test <file/dir> | --batch <dir> | <file>")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()