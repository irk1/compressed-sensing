import os
import sys
import gc
import struct
import json
import time
import signal
import psutil
import rawpy
import datetime
import argparse
import multiprocessing
import numpy as np
import cv2
import cvxpy as cp
import tifffile
from multiprocessing import sharedctypes

# =================================================================
# 1. HARDWARE, SIGNALS & LOGGING
# =================================================================
os.environ["CVXPY_SOLVER_MAP_IGNORE"] = "OSQP"
KEEP_RUNNING = True
SETTINGS_FILE = "settings.json"

def graceful_exit(signum, frame):
    global KEEP_RUNNING
    print("\n[!] STOP SIGNAL RECEIVED. Finishing current tiles and saving...")
    KEEP_RUNNING = False

signal.signal(signal.SIGINT, graceful_exit)

class LiveLogger:
    def __init__(self, filename="training_results.csv"):
        self.filename = filename
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', encoding="utf-8") as f:
                f.write("Timestamp_UTC,File_Name,Lambda,TV_Weight,Duration_Sec,Status\n")
                f.flush()
                os.fsync(f.fileno())

    def log_entry(self, file_name, l_val, t_val, duration, status="SUCCESS"):
        utc_now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        line = f"{utc_now},{file_name},{l_val},{t_val},{duration:.2f},{status}\n"
        with open(self.filename, 'a', encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

logger = LiveLogger()

class Silence:
    def __init__(self):
        self.null_fd = os.open(os.devnull, os.O_RDWR)
        self.save_fds = (os.dup(1), os.dup(2))
    def __enter__(self):
        os.dup2(self.null_fd, 1); os.dup2(self.null_fd, 2)
    def __exit__(self, *_):
        os.dup2(self.save_fds[0], 1); os.dup2(self.save_fds[1], 2)
        os.close(self.null_fd); os.close(self.save_fds[0]); os.close(self.save_fds[1])

# =================================================================
# 2. DNG WRITER ENGINE (v4.2 - DNG 1.7.1.0 Compliant)
# =================================================================
class Tag:
    NewSubfileType = (254, 4); ImageWidth = (256, 4); ImageLength = (257, 4)
    BitsPerSample = (258, 3); Compression = (259, 3); PhotometricInterpretation = (262, 3)
    SamplesPerPixel = (277, 3); Software = (305, 2); StripOffsets = (273, 4)
    StripByteCounts = (279, 4); Orientation = (274, 3); DNGVersion = (50706, 1)
    DNGBackwardVersion = (50707, 1); UniqueCameraModel = (50708, 2)
    ColorMatrix1 = (50721, 10); AsShotNeutral = (50728, 5)
    CalibrationIlluminant1 = (50778, 3); BlackLevel = (50714, 5); WhiteLevel = (50717, 3)

class dngTag:
    def __init__(self, tag_def, value):
        self.TagId, self.DataType = tag_def
        self.DataCount = len(value) if not isinstance(value, (str, bytes)) else len(value) + 1
        self.packed_val = self._pack_raw(value)
        self.is_inline = len(self.packed_val) <= 4

    def _pack_raw(self, val):
        if self.DataType == 1: return struct.pack(f"<{len(val)}B", *val)
        if self.DataType == 2: return val.encode('ascii') + b'\x00'
        if self.DataType == 3: return struct.pack(f"<{len(val)}H", *val)
        if self.DataType == 4: return struct.pack(f"<{len(val)}L", *val)
        if self.DataType == 5: return struct.pack(f"<{len(val)*2}L", *[i for sub in val for i in sub])
        if self.DataType == 10: return struct.pack(f"<{len(val)*2}l", *[i for sub in val for i in sub])
        return b''

    def write(self, buf, tag_off, heap_off):
        if self.is_inline:
            val_bytes = self.packed_val.ljust(4, b'\x00')
            inline_val = struct.unpack("<I", val_bytes)[0]
            struct.pack_into("<HHII", buf, tag_off, self.TagId, self.DataType, self.DataCount, inline_val)
            return heap_off
        else:
            struct.pack_into("<HHII", buf, tag_off, self.TagId, self.DataType, self.DataCount, heap_off)
            buf[heap_off:heap_off+len(self.packed_val)] = self.packed_val
            return heap_off + (len(self.packed_val) + 3) & ~3 

class MonolithDNGWriter:
    def save(self, data, path):
        h, w, chans = data.shape
        pixel_data = data.tobytes()
        
        tags = [
            dngTag(Tag.NewSubfileType, [0]), dngTag(Tag.ImageWidth, [w]), dngTag(Tag.ImageLength, [h]),
            dngTag(Tag.BitsPerSample, [16] * chans), dngTag(Tag.Compression, [1]),
            dngTag(Tag.PhotometricInterpretation, [34892]), dngTag(Tag.SamplesPerPixel, [chans]),
            dngTag(Tag.Software, "Irk_Monolith_v5.2_Xeon"), 
            dngTag(Tag.DNGVersion, [1, 7, 1, 0]),
            dngTag(Tag.DNGBackwardVersion, [1, 6, 0, 0]),
            dngTag(Tag.Orientation, [1]), dngTag(Tag.UniqueCameraModel, "Xeon_Botanical_Custom"),
            dngTag(Tag.BlackLevel, [(0, 1)] * chans), 
            dngTag(Tag.WhiteLevel, [65535] * chans), 
            dngTag(Tag.AsShotNeutral, [(1, 1)] * chans), 
            dngTag(Tag.CalibrationIlluminant1, [21]),
            dngTag(Tag.ColorMatrix1, [(1,1), (0,1), (0,1), (0,1), (1,1), (0,1), (0,1), (0,1), (1,1)])
        ]
        
        ifd_start = 16
        heap_start = (ifd_start + 2 + (len(tags) + 2) * 12 + 4 + 255) & ~255
        temp_heap = heap_start
        for t in tags:
            if not t.is_inline: temp_heap = (temp_heap + len(t.packed_val) + 3) & ~3
        
        pixel_off = (temp_heap + 1024) & ~255
        tags.append(dngTag(Tag.StripOffsets, [pixel_off]))
        tags.append(dngTag(Tag.StripByteCounts, [len(pixel_data)]))
        tags.sort(key=lambda x: x.TagId)

        header = bytearray(pixel_off)
        struct.pack_into("<ccbbI", header, 0, b"I", b"I", 42, 0, ifd_start)
        struct.pack_into("<H", header, ifd_start, len(tags))
        
        tp, hp = ifd_start + 2, heap_start
        for t in tags:
            hp = t.write(header, tp, hp)
            tp += 12
            
        with open(path, "wb") as f:
            f.write(header)
            f.write(pixel_data)

# =================================================================
# 3. MATH KERNEL & LOADER
# =================================================================
def load_image(path):
    valid_raw = ('.dng', '.nef', '.cr2', '.arw', '.orf')
    try:
        if path.lower().endswith(valid_raw):
            with rawpy.imread(path) as raw:
                img_float = raw.postprocess(
                    use_camera_wb=True, 
                    output_bps=16, 
                    gamma=(1,1), 
                    no_auto_bright=True
                ).astype(np.float32) / 65535.0
                return img_float
        else:
            bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if bgr is None: return None
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return rgb.astype(np.float32) / (255.0 if rgb.dtype == np.uint8 else 65535.0)
    except Exception as e:
        print(f"[ERROR] Loading Failed: {e}")
        return None

def get_dct_matrix(n):
    d = np.zeros((n, n))
    freq = np.pi * (np.arange(n) + 0.5) / n
    for k in range(n):
        d[k, :] = np.sqrt(1/n) if k == 0 else np.sqrt(2/n) * np.cos(k * freq)
    return d

def solve_tile_shared(args):
    y_idx, x_idx, ts, scale, l_min, tv_w, t_id, shm_shape = args
    shared_arr = np.frombuffer(shared_mem_base, dtype=np.float32).reshape(shm_shape)
    lr_tile = shared_arr[y_idx:y_idx+ts, x_idx:x_idx+ts]
    hh, ww = ts * scale, ts * scale
    D_h, D_w = get_dct_matrix(hh), get_dct_matrix(ww)
    X_dct = cp.Variable((hh, ww))
    img_rec = D_h.T @ X_dct @ D_w
    constraints = [cp.abs(img_rec[::scale, ::scale] - lr_tile) <= 0.005, img_rec >= 0, img_rec <= 1]
    prob = cp.Problem(cp.Minimize((cp.norm(X_dct, 1) * l_min) + (cp.tv(img_rec) * tv_w)), constraints)
    try:
        with Silence(): prob.solve(solver=cp.ECOS, max_iters=500)
        if X_dct.value is None: raise ValueError
        return (t_id, np.clip(img_rec.value, 0, 1).astype(np.float32))
    except:
        return (t_id, cv2.resize(lr_tile, (ww, hh), interpolation=cv2.INTER_LANCZOS4))

# =================================================================
# 4. ORCHESTRATION & GENERATORS
# =================================================================
def init_worker(shm):
    global shared_mem_base
    shared_mem_base = shm

def work_generator(H, W, ts, stride, scale, l_min, tv_w, chan_shape, config):
    max_ram_bytes = config.get("MAX_RAM_GB", 30) * (1024**3)
    tile_count = 0
    for y in range(0, H - ts + 1, stride):
        for x in range(0, W - ts + 1, stride):
            while psutil.virtual_memory().available < (psutil.virtual_memory().total - max_ram_bytes):
                time.sleep(0.5)
            if not KEEP_RUNNING:
                return
            yield (y, x, ts, scale, l_min, tv_w, tile_count, chan_shape)
            tile_count += 1

def process_full(img_float, config, l_override=None, t_override=None):
    l_min = l_override if l_override is not None else config["LAMBDA_MIN"]
    tv_w = t_override if t_override is not None else config["TV_WEIGHT"]

    lab = cv2.cvtColor(img_float, cv2.COLOR_RGB2Lab)
    H, W, _ = lab.shape
    scale = config["SCALE_FACTOR"]
    ts, overlap = config["TILE_SIZE"], config["OVERLAP"]
    stride = ts - overlap
    
    up_chans = []
    for c_idx in range(3):
        chan_data = lab[:, :, c_idx].copy()
        shm = sharedctypes.RawArray('f', chan_data.size)
        np.frombuffer(shm, dtype=np.float32).reshape(chan_data.shape)[:] = chan_data
        
        out = np.zeros((H * scale, W * scale), dtype=np.float32)
        cpus = min(multiprocessing.cpu_count(), config["MAX_THREADS"])
        
        pos = {}
        tile_cnt = 0
        for y in range(0, H - ts + 1, stride):
            for x in range(0, W - ts + 1, stride):
                pos[tile_cnt] = (y*scale, x*scale)
                tile_cnt += 1

        work_gen = work_generator(H, W, ts, stride, scale, l_min, tv_w, chan_data.shape, config)
        
        with multiprocessing.Pool(processes=cpus, initializer=init_worker, initargs=(shm,)) as pool:
            for t_id, result in pool.imap_unordered(solve_tile_shared, work_gen, chunksize=4):
                y_up, x_up = pos[t_id]
                out[y_up:y_up+(ts*scale), x_up:x_up+(ts*scale)] = result
                
        up_chans.append(out); del chan_data; gc.collect()

    final_rgb = cv2.cvtColor(cv2.merge(up_chans), cv2.COLOR_Lab2RGB)
    
    if config.get("DENOISE", False):
        rgb_8 = (np.clip(final_rgb, 0, 1) * 255).astype(np.uint8)
        bgr_8 = cv2.cvtColor(rgb_8, cv2.COLOR_RGB2BGR)
        cleaned_8 = cv2.fastNlMeansDenoisingColored(bgr_8, None, 3, 3, 7, 21)
        cleaned_rgb = cv2.cvtColor(cleaned_8, cv2.COLOR_BGR2RGB)
        return (cleaned_rgb.astype(np.uint16) * 257)
    
    return (np.clip(final_rgb, 0, 1) * 65535).astype(np.uint16)

# =================================================================
# 5. AUTO-TUNER & SCORING METRICS (WITH PARTIAL SAVING)
# =================================================================
def score_image_quality(img_16bit):
    img_8bit = (img_16bit / 257).astype(np.uint8)
    gray = cv2.cvtColor(img_8bit, cv2.COLOR_RGB2GRAY)
    
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blurred = cv2.medianBlur(gray, 3)
    noise_var = np.var(gray.astype(np.float64) - blurred.astype(np.float64))
    
    if noise_var < 1e-5: noise_var = 1e-5
    return laplacian_var / noise_var

def run_auto_tuner(folder_path, config, valid_exts, use_full_image=False):
    print(f"[*] AUTO-TUNER INITIATED: Analyzing folder '{folder_path}'")
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]
    
    lambdas = [1e-6, 1e-5, 1e-4]
    tvs = [0.001, 0.002, 0.005]
    
    score_board = {(l, t): [] for l in lambdas for t in tvs}
    images_completed = 0
    
    for f in files:
        if not KEEP_RUNNING: 
            print("\n[!] Tuning interrupted. Calculating results from completed images...")
            break
            
        img_float = load_image(f)
        if img_float is None: continue
        
        if use_full_image:
            print(f"  [>] Processing FULL image for {os.path.basename(f)}...")
            target_data = img_float
        else:
            h, w, _ = img_float.shape
            cy, cx = h // 2, w // 2
            crop_size = min(512, h, w)
            half_crop = crop_size // 2
            target_data = img_float[cy-half_crop:cy+half_crop, cx-half_crop:cx+half_crop]
            print(f"  [>] Processing central {crop_size}x{crop_size} crop for {os.path.basename(f)}...")
        
        for l_val in lambdas:
            for t_val in tvs:
                if not KEEP_RUNNING: break
                res_data = process_full(target_data, config, l_val, t_val)
                score = score_image_quality(res_data)
                score_board[(l_val, t_val)].append(score)
        
        if KEEP_RUNNING:
            images_completed += 1
                
    if images_completed == 0 and not score_board[(lambdas[0], tvs[0])]:
        print("\n[!] Tuning aborted before any scores could be calculated. Settings not updated.")
        return
    
    print(f"\n[*] AUTO-TUNING RESULTS (Based on {images_completed} completed images):")
    best_combo = None
    best_score = -1
    
    for (l_val, t_val), scores in score_board.items():
        if not scores: continue
        avg_score = np.mean(scores)
        print(f"    Lambda: {l_val}, TV: {t_val} -> Score: {avg_score:.2f}")
        if avg_score > best_score:
            best_score = avg_score
            best_combo = (l_val, t_val)
            
    if best_combo:
        print(f"\n[+] WINNER FOUND: Lambda={best_combo[0]}, TV={best_combo[1]}")
        config["LAMBDA_MIN"] = best_combo[0]
        config["TV_WEIGHT"] = best_combo[1]
        with open(SETTINGS_FILE, "w") as f:
            json.dump(config, f, indent=4)
        print(f"[+] Updated '{SETTINGS_FILE}' with optimal parameters. Ready for batch processing!")

# =================================================================
# 6. EXECUTION MODES
# =================================================================
def main():
    multiprocessing.freeze_support()
    try: multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError: pass

    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", help="Path to RAW file or folder")
    parser.add_argument("--test", help="Run 9-grid test mode on a file or folder")
    parser.add_argument("--batch", help="Run resumable batch processing on a folder")
    parser.add_argument("--tune", help="Auto-tune settings using a folder of sample images")
    parser.add_argument("--full", action="store_true", help="Use full images instead of center crops for auto-tuning")
    parser.add_argument("--dng", action="store_true", help="Output as DNG instead of TIF")
    args = parser.parse_args()

    cfg = {"SCALE_FACTOR": 2, "TILE_SIZE": 64, "OVERLAP": 8, "LAMBDA_MIN": 1e-4, "TV_WEIGHT": 0.005, "MAX_THREADS": 48, "MAX_RAM_GB": 30, "DENOISE": False}
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f: cfg.update(json.load(f))
        
    valid_exts = ('.dng', '.arw', '.cr2', '.nef', '.orf', '.jpg', '.png', '.tif')

    if args.tune:
        run_auto_tuner(args.tune, cfg, valid_exts, use_full_image=args.full)

    elif args.test:
        path = args.test
        files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(valid_exts)] if os.path.isdir(path) else [path]
        for f in files:
            if not KEEP_RUNNING: break
            img = load_image(f)
            if img is not None: 
                print(f"[!] TEST GRID: Generating 9 variations for {os.path.basename(f)}...")
                for l_val in [1e-6, 1e-5, 1e-4]:
                    for t_val in [0.001, 0.002, 0.005]:
                        if not KEEP_RUNNING: break
                        print(f"  [>] Solving: Lambda={l_val}, TV={t_val}")
                        res = process_full(img, cfg, l_val, t_val)
                        out_path = f"{os.path.splitext(f)[0]}_L{l_val}_T{t_val}.{'dng' if args.dng else 'tif'}"
                        if args.dng: MonolithDNGWriter().save(res, out_path)
                        else: tifffile.imwrite(out_path, res, compression=32946)

    elif args.batch:
        batch_dir = args.batch
        files = sorted([os.path.join(batch_dir, f) for f in os.listdir(batch_dir) if f.lower().endswith(valid_exts)])
        for f in files:
            if not KEEP_RUNNING: break
            out_path = f"{os.path.splitext(f)[0]}_irk_v52.{'dng' if args.dng else 'tif'}"
            if os.path.exists(out_path): 
                print(f"[*] Skipping {os.path.basename(f)} (Already processed)")
                continue
            
            print(f"[*] BATCH RUN: {os.path.basename(f)}")
            start_time = time.time()
            img = load_image(f)
            if img is not None:
                res = process_full(img, cfg)
                if args.dng: MonolithDNGWriter().save(res, out_path)
                else: tifffile.imwrite(out_path, res, compression=32946)
                logger.log_entry(os.path.basename(f), cfg["LAMBDA_MIN"], cfg["TV_WEIGHT"], time.time()-start_time, "SUCCESS")
                print(f"[COMPLETE] Saved: {out_path}")

    elif args.input:
        path = args.input
        files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(valid_exts)] if os.path.isdir(path) else [path]
        for f in files:
            if not KEEP_RUNNING: break
            start_time = time.time()
            img = load_image(f)
            if img is not None:
                res = process_full(img, cfg)
                out_path = f"{os.path.splitext(f)[0]}_irk_v52.{'dng' if args.dng else 'tif'}"
                if args.dng: MonolithDNGWriter().save(res, out_path)
                else: tifffile.imwrite(out_path, res, compression=32946)
                logger.log_entry(os.path.basename(f), cfg["LAMBDA_MIN"], cfg["TV_WEIGHT"], time.time()-start_time, "SUCCESS")
                print(f"[COMPLETE] Saved: {out_path}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()