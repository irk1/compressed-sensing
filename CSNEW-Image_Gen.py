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
CAMERA_METADATA = {}

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
# 2. DNG WRITER ENGINE (v5.3.5 - HYBRID READY)
# =================================================================
class Tag:
    NewSubfileType = (254, 4); ImageWidth = (256, 4); ImageLength = (257, 4)
    BitsPerSample = (258, 3); Compression = (259, 3); PhotometricInterpretation = (262, 3)
    SamplesPerPixel = (277, 3); Software = (305, 2); StripOffsets = (273, 4)
    StripByteCounts = (279, 4); Orientation = (274, 3); DNGVersion = (50706, 1)
    DNGBackwardVersion = (50707, 1); UniqueCameraModel = (50708, 2)
    ColorMatrix1 = (50721, 10); AsShotNeutral = (50728, 5)
    CalibrationIlluminant1 = (50778, 3); BlackLevel = (50714, 4); WhiteLevel = (50717, 4)
    BlackLevelRepeatDim = (50713, 3); CFARepeatPatternDim = (33421, 3); CFAPattern = (33422, 1)

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
        h, w = data.shape
        pixel_data = data.tobytes()
        meta = CAMERA_METADATA if CAMERA_METADATA else {"pattern": [0,1,1,2], "black": [0,0,0,0], "white": 65535}
        
        tags = [
            dngTag(Tag.NewSubfileType, [0]), dngTag(Tag.ImageWidth, [w]), dngTag(Tag.ImageLength, [h]),
            dngTag(Tag.BitsPerSample, [16]), dngTag(Tag.Compression, [1]),
            dngTag(Tag.PhotometricInterpretation, [32803]), 
            dngTag(Tag.SamplesPerPixel, [1]),
            dngTag(Tag.Software, "Irk_Monolith_v5.3.5_Hybrid"), 
            dngTag(Tag.DNGVersion, [1, 4, 0, 0]), 
            dngTag(Tag.DNGBackwardVersion, [1, 3, 0, 0]), 
            dngTag(Tag.Orientation, [1]), dngTag(Tag.UniqueCameraModel, "Xeon_Botanical_Custom"),
            dngTag(Tag.CFARepeatPatternDim, [2, 2]),
            dngTag(Tag.CFAPattern, meta["pattern"]),
            dngTag(Tag.BlackLevelRepeatDim, [2, 2]),
            dngTag(Tag.BlackLevel, meta["black"]), 
            dngTag(Tag.WhiteLevel, [meta["white"]]),
            dngTag(Tag.AsShotNeutral, meta.get("as_shot_neutral", [(1, 1), (1, 1), (1, 1)])),
            dngTag(Tag.CalibrationIlluminant1, [21]),
            dngTag(Tag.ColorMatrix1, meta.get("color_matrix", [(1,1), (0,1), (0,1), (0,1), (1,1), (0,1), (0,1), (0,1), (1,1)]))
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
# 3. MATH KERNEL & LOADING
# =================================================================
def load_image(path):
    global CAMERA_METADATA
    ext = os.path.splitext(path)[1].lower()
    
    # 1. Standard RGB Image Pipeline
    if ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']:
        try:
            if ext in ['.tif', '.tiff']:
                img = tifffile.imread(path)
            else:
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is not None and len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if img is None: return None, False
            
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif len(img.shape) == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
                
            if img.dtype == np.uint8: img = img.astype(np.float32) / 255.0
            elif img.dtype == np.uint16: img = img.astype(np.float32) / 65535.0
            else: img = img.astype(np.float32)
            
            return img, False # False = Not CFA/RAW
        except Exception as e:
            print(f"[ERROR] Standard Image Loading Failed: {e}")
            return None, False

    # 2. True RAW / CFA Pipeline
    try:
        with rawpy.imread(path) as raw:
            bayer = raw.raw_image_visible.astype(np.float32)
            white_level = float(max(raw.camera_white_level_per_channel))
            bayer = bayer / white_level
            
            h, w = bayer.shape
            cfa_packed = np.zeros((h//2, w//2, 4), dtype=np.float32)
            cfa_packed[:, :, 0] = bayer[0::2, 0::2]
            cfa_packed[:, :, 1] = bayer[0::2, 1::2]
            cfa_packed[:, :, 2] = bayer[1::2, 0::2]
            cfa_packed[:, :, 3] = bayer[1::2, 1::2]
            
            pattern_raw = raw.raw_pattern.flatten().tolist()
            pattern_dng = [1 if x == 3 else x for x in pattern_raw]
            
            wb = raw.camera_whitebalance
            nr = int(10000 / wb[0]) if wb[0] > 0 else 10000
            ng = int(10000 / wb[1]) if wb[1] > 0 else 10000
            nb = int(10000 / wb[2]) if wb[2] > 0 else 10000

            try:
                xyz_cam = raw.rgb_xyz_matrix[:3, :3]
                if np.sum(np.abs(xyz_cam)) < 0.1: xyz_cam = np.eye(3)
            except:
                xyz_cam = np.eye(3)
                
            c_mat = [(int(val * 10000), 10000) for val in xyz_cam.flatten()]
            
            CAMERA_METADATA = {
                "pattern": pattern_dng,
                "black": [int(x) for x in raw.black_level_per_channel],
                "white": int(white_level),
                "as_shot_neutral": [(nr, 10000), (ng, 10000), (nb, 10000)],
                "color_matrix": c_mat
            }
            return cfa_packed, True # True = Is CFA/RAW
    except Exception as e:
        print(f"[ERROR] RAW Loading Failed: {e}")
        return None, False

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
            if not KEEP_RUNNING: return
            yield (y, x, ts, scale, l_min, tv_w, tile_count, chan_shape)
            tile_count += 1

def process_full(img_float, is_cfa, config, l_override=None, t_override=None):
    l_min = l_override if l_override is not None else config["LAMBDA_MIN"]
    tv_w = t_override if t_override is not None else config["TV_WEIGHT"]

    H, W, chans = img_float.shape
    scale = config["SCALE_FACTOR"]
    ts, overlap = config["TILE_SIZE"], config["OVERLAP"]
    stride = ts - overlap
    
    up_chans = []
    for c_idx in range(chans):
        chan_data = img_float[:, :, c_idx].copy()
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
        up_chans.append(out); gc.collect()

    if is_cfa:
        out_h, out_w = H * scale * 2, W * scale * 2
        final_bayer = np.zeros((out_h, out_w), dtype=np.float32)
        final_bayer[0::2, 0::2] = up_chans[0]
        final_bayer[0::2, 1::2] = up_chans[1]
        final_bayer[1::2, 0::2] = up_chans[2]
        final_bayer[1::2, 1::2] = up_chans[3]
        
        white_level = CAMERA_METADATA.get("white", 65535)
        return (np.clip(final_bayer, 0, 1) * white_level).astype(np.uint16)
    else:
        out_h, out_w = H * scale, W * scale
        final_rgb = np.zeros((out_h, out_w, chans), dtype=np.float32)
        for i in range(chans):
            final_rgb[:, :, i] = up_chans[i]
        return np.clip(final_rgb, 0, 1)

# =================================================================
# 5. AUTO-TUNER (PLACEHOLDER FOR CFA)
# =================================================================
def run_auto_tuner(folder_path, config, valid_exts, use_full_image=False):
    print("[!] Auto-Tuner is currently optimized for RGB space. Using manual settings from JSON.")
    return

# =================================================================
# 6. EXECUTION MODES
# =================================================================
def main():
    multiprocessing.freeze_support()
    try: multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError: pass

    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", help="Path to RAW file or folder")
    parser.add_argument("--test", help="Run 9-grid test mode on a file")
    parser.add_argument("--batch", help="Run resumable batch processing")
    parser.add_argument("--dng", action="store_true", help="Output as DNG (True RAW)")
    args = parser.parse_args()

    cfg = {"SCALE_FACTOR": 2, "TILE_SIZE": 64, "OVERLAP": 16, "LAMBDA_MIN": 1e-4, "TV_WEIGHT": 0.005, "MAX_THREADS": 48, "MAX_RAM_GB": 30}
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f: cfg.update(json.load(f))
        
    valid_exts = ('.dng', '.arw', '.cr2', '.nef', '.orf', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')

    if args.test:
        f = args.test
        img, is_cfa = load_image(f)
        if img is not None:
            for l_val in [1e-6, 1e-5, 1e-4]:
                for t_val in [0.001, 0.002, 0.005]:
                    if not KEEP_RUNNING: break
                    res = process_full(img, is_cfa, cfg, l_val, t_val)
                    if is_cfa:
                        out = f"{os.path.splitext(f)[0]}_L{l_val}_T{t_val}.dng"
                        MonolithDNGWriter().save(res, out)
                    else:
                        out = f"{os.path.splitext(f)[0]}_L{l_val}_T{t_val}.tif"
                        tifffile.imwrite(out, (res * 65535.0).astype(np.uint16))

    elif args.batch or args.input:
        path = args.batch if args.batch else args.input
        files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(valid_exts)] if os.path.isdir(path) else [path]
        for f in files:
            if not KEEP_RUNNING: break
            
            img, is_cfa = load_image(f)
            if img is not None:
                if is_cfa:
                    out_path = f"{os.path.splitext(f)[0]}_irk_v57.dng"
                else:
                    out_path = f"{os.path.splitext(f)[0]}_irk_v57.tif"
                    
                if os.path.exists(out_path): continue
                
                start_time = time.time()
                res = process_full(img, is_cfa, cfg)
                
                if is_cfa:
                    MonolithDNGWriter().save(res, out_path)
                else:
                    tifffile.imwrite(out_path, (res * 65535.0).astype(np.uint16))
                    
                logger.log_entry(os.path.basename(f), cfg["LAMBDA_MIN"], cfg["TV_WEIGHT"], time.time()-start_time)
                print(f"[COMPLETE] Saved: {out_path}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()