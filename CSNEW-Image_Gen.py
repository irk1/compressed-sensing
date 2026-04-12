import os
import sys
import gc
import struct
import json
import rawpy
import multiprocessing
import argparse
import numpy as np
import cv2
import cvxpy as cp
import tifffile
from multiprocessing import sharedctypes

# =================================================================
# 1. DEEP SYSTEM SILENCER
# =================================================================
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
    
    # FIX: CalibrationIlluminant1 ID changed from 50723 to 50778
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
            dngTag(Tag.Software, "Irk_Monolith_v4.2_Xeon"), 
            dngTag(Tag.DNGVersion, [1, 7, 1, 0]), # DNG 1.7.1.0
            dngTag(Tag.DNGBackwardVersion, [1, 6, 0, 0]), # Backward Compat
            dngTag(Tag.Orientation, [1]), dngTag(Tag.UniqueCameraModel, "Xeon_Botanical_Custom"),
            dngTag(Tag.BlackLevel, [(0, 1)] * chans), 
            dngTag(Tag.WhiteLevel, [65535] * chans), # Match SamplesPerPixel 
            dngTag(Tag.AsShotNeutral, [(1, 1)] * chans), 
            dngTag(Tag.CalibrationIlluminant1, [21]), # Illuminant D65 (Type 3) 
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
# 3. MATH KERNEL (Compressed Sensing)
# =================================================================
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
# 4. ORCHESTRATION
# =================================================================
def init_worker(shm):
    global shared_mem_base
    shared_mem_base = shm

def process_full(img_path, config, use_dng=False):
    with rawpy.imread(img_path) as raw:
        img_float = raw.postprocess(use_camera_wb=True, output_bps=16).astype(np.float32) / 65535.0
    lab = cv2.cvtColor(img_float, cv2.COLOR_RGB2Lab)
    H, W, _ = lab.shape
    scale = config["SCALE_FACTOR"]
    up_chans = []
    for c_idx in range(3):
        chan_data = lab[:, :, c_idx].copy()
        shm = sharedctypes.RawArray('f', chan_data.size)
        np.frombuffer(shm, dtype=np.float32).reshape(chan_data.shape)[:] = chan_data
        ts, overlap = config["TILE_SIZE"], config["OVERLAP"]
        stride = ts - overlap
        work, pos, tile_count = [], {}, 0
        for y in range(0, H - ts + 1, stride):
            for x in range(0, W - ts + 1, stride):
                work.append((y, x, ts, scale, config["LAMBDA_MIN"], config["TV_WEIGHT"], tile_count, chan_data.shape))
                pos[tile_count] = (y*scale, x*scale)
                tile_count += 1
        out = np.zeros((H * scale, W * scale), dtype=np.float32)
        cpus = min(multiprocessing.cpu_count(), config["MAX_THREADS"])
        with multiprocessing.Pool(processes=cpus, initializer=init_worker, initargs=(shm,)) as pool:
            for t_id, result in pool.imap_unordered(solve_tile_shared, work, chunksize=4):
                y_up, x_up = pos[t_id]
                out[y_up:y_up+(ts*scale), x_up:x_up+(ts*scale)] = result
        up_chans.append(out); del chan_data; gc.collect()
    final_rgb = cv2.cvtColor(cv2.merge(up_chans), cv2.COLOR_Lab2RGB)
    data_to_save = (np.clip(final_rgb, 0, 1) * 65535).astype(np.uint16)
    out_path = f"{os.path.splitext(img_path)[0]}_irk_v42.{'dng' if use_dng else 'tif'}"
    if use_dng: MonolithDNGWriter().save(data_to_save, out_path)
    else: tifffile.imwrite(out_path, data_to_save)
    print(f"[COMPLETE] {out_path}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    try: multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError: pass
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to RAW file or folder")
    parser.add_argument("--dng", action="store_true")
    args = parser.parse_args()
    cfg = {"SCALE_FACTOR": 2, "TILE_SIZE": 64, "OVERLAP": 8, "LAMBDA_MIN": 1e-4, "TV_WEIGHT": 0.005, "MAX_THREADS": 48}
    if os.path.exists("settings.json"):
        with open("settings.json", "r") as f: cfg.update(json.load(f))
    valid_exts = ('.dng', '.arw', '.cr2', '.nef', '.orf')
    files = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith(valid_exts)] if os.path.isdir(args.input) else [args.input]
    for f in files: process_full(f, cfg, use_dng=args.dng)