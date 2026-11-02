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
import zlib
from multiprocessing import sharedctypes
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import correlation


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
                f.write("Date_UTC,Time_UTC,File_Name,Scale_Factor,Lambda,TV_Weight,Duration_Sec,MSE,PSNR,SSIM,Correlation,Composite_Loss\n")
                f.flush()
                os.fsync(f.fileno())

    def log_entry(self, file_name, scale, l_val, t_val, duration, mse=None, psnr=None, ssim=None, corr=None, comp_loss=None):
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        date_utc = now_utc.strftime('%Y-%m-%d')
        time_utc = now_utc.strftime('%H:%M:%S')        
        scale_str = str(scale) if scale is not None else "N/A"
        m_str = f"{mse:.6f}" if mse is not None else "N/A"
        p_str = f"{psnr:.4f}" if psnr is not None else "N/A"
        s_str = f"{ssim:.6f}" if ssim is not None else "N/A"
        c_str = f"{corr:.6f}" if corr is not None else "N/A"
        l_str = f"{comp_loss:.6f}" if comp_loss is not None else "N/A"
        
        line = f"{date_utc},{time_utc},{file_name},{scale_str},{l_val},{t_val},{duration:.2f},{m_str},{p_str},{s_str},{c_str},{l_str}\n"
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
# 2. DNG WRITER ENGINE (v5.7.1 - COMPLIANCE READY)
# =================================================================
class Tag:
    NewSubfileType = (254, 4); ImageWidth = (256, 4); ImageLength = (257, 4)
    BitsPerSample = (258, 3); Compression = (259, 3); PhotometricInterpretation = (262, 3)
    PlanarConfiguration = (284, 3); SamplesPerPixel = (277, 3); Software = (305, 2)
    StripOffsets = (273, 4); StripByteCounts = (279, 4); Orientation = (274, 3)
    DNGVersion = (50706, 1); DNGBackwardVersion = (50707, 1); UniqueCameraModel = (50708, 2)
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
        # Apply bitwise masks to bypass Python's strict sign checking.
        # This guarantees the exact binary bytes are written regardless of positive/negative.
        if self.DataType == 1: 
            return struct.pack(f"<{len(val)}B", *[v & 0xFF for v in val])
        
        if self.DataType == 2: 
            return val.encode('ascii') + b'\x00'
            
        if self.DataType == 3: 
            return struct.pack(f"<{len(val)}H", *[v & 0xFFFF for v in val])
            
        if self.DataType == 4: 
            return struct.pack(f"<{len(val)}I", *[v & 0xFFFFFFFF for v in val])
            
        if self.DataType == 5: 
            flat_val = [i for sub in val for i in sub]
            return struct.pack(f"<{len(flat_val)}I", *[v & 0xFFFFFFFF for v in flat_val])
            
        if self.DataType == 10: 
            flat_val = [i for sub in val for i in sub]
            # 10 is strictly SRATIONAL (Signed), so we can safely use 'i' here
            return struct.pack(f"<{len(flat_val)}i", *flat_val)
            
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
    
    def save(self, data, path, is_cfa=True):
        meta = CAMERA_METADATA if CAMERA_METADATA else {"pattern": [0,1,1,2], "black": [0,0,0,0], "white": 65535}
        
        # FAILSAFE 1: Strip 4th channel (Alpha/Padding) for Linear DNGs to strictly match 3x3 Color Matrix
        if len(data.shape) == 3 and data.shape[2] > 3:
            data = data[:, :, :3]

        # FAILSAFE 2: Convert 32-bit floats to 16-bit unsigned integers
        if data.dtype != np.uint16:
            data = (np.clip(data, 0, 1) * meta["white"]).astype(np.uint16)

        if len(data.shape) == 2:
            h, w = data.shape
            samples, photo_interp = 1, 32803 # True RAW (CFA)
        else:
            h, w, samples = data.shape
            photo_interp = 34892 # Linear RAW (RGB Demosaiced)

        # --- COMPRESSION IMPLEMENTATION ---
        raw_bytes = data.tobytes()
        print(f"[*] Compressing {len(raw_bytes)/(1024**3):.2f} GB of DNG data to bypass 32-bit limits...")
        # zlib level 6 provides a great balance of speed vs file size reduction
        compressed_data = zlib.compress(raw_bytes, level=6) 
        
        # Handle BlackLevel channels
        black_level = meta["black"]
        if samples == 3:
            black_level = [black_level[0]] * 3 if len(black_level) > 0 else [0, 0, 0]

        tags = [
            dngTag(Tag.NewSubfileType, [0]), dngTag(Tag.ImageWidth, [w]), dngTag(Tag.ImageLength, [h]),
            dngTag(Tag.BitsPerSample, [16] * samples), 
            dngTag(Tag.Compression, [8]), # CHANGED: 8 = Deflate Compression
            dngTag(Tag.PhotometricInterpretation, [photo_interp]), 
            dngTag(Tag.SamplesPerPixel, [samples]),
            dngTag(Tag.Software, "Irk_Monolith_v5.7.2"), 
            dngTag(Tag.DNGVersion, [1, 4, 0, 0]), 
            dngTag(Tag.DNGBackwardVersion, [1, 4, 0, 0]), 
            dngTag(Tag.Orientation, [1]), dngTag(Tag.UniqueCameraModel, "Xeon_Botanical_Custom"),
            dngTag(Tag.WhiteLevel, [meta["white"]]),
            dngTag(Tag.CalibrationIlluminant1, [21]),
            dngTag(Tag.ColorMatrix1, meta.get("color_matrix", [(1,1), (0,1), (0,1), (0,1), (1,1), (0,1), (0,1), (0,1), (1,1)])),
            dngTag(Tag.BlackLevel, black_level),
            dngTag(Tag.AsShotNeutral, meta.get("as_shot_neutral", [(1, 1), (1, 1), (1, 1)]))
        ]
        
        # Require Chunky Planar Configuration for RGB data
        if samples > 1:
            tags.append(dngTag(Tag.PlanarConfiguration, [1]))
        
        # ONLY apply Bayer Pattern tags if it is actually a Bayer image
        if is_cfa or photo_interp == 32803:
            tags.extend([
                dngTag(Tag.CFARepeatPatternDim, [2, 2]),
                dngTag(Tag.CFAPattern, meta["pattern"]),
                dngTag(Tag.BlackLevelRepeatDim, [2, 2])
            ])
            
        tags.sort(key=lambda x: x.TagId)
        
        ifd_start = 16
        heap_start = (ifd_start + 2 + (len(tags) + 2) * 12 + 4 + 255) & ~255
        temp_heap = heap_start
        for t in tags:
            if not t.is_inline: temp_heap = (temp_heap + len(t.packed_val) + 3) & ~3
        
        pixel_off = (temp_heap + 1024) & ~255
        tags.append(dngTag(Tag.StripOffsets, [pixel_off]))
        # CHANGED: Register the much smaller compressed byte count
        tags.append(dngTag(Tag.StripByteCounts, [len(compressed_data)]))
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
            # CHANGED: Write the compressed data block
            f.write(compressed_data)

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

    # 2. RAW / CFA & Linear DNG Pipeline
    try:
        c_mat, as_shot_neutral = None, None
        try:
            # Extracts exact values with original signs (no masks, no absolute values)
            def extract_exact_rational(v):
                if hasattr(v, 'numerator'): 
                    return (int(v.numerator), int(v.denominator))
                elif isinstance(v, tuple) and len(v) == 2: 
                    return (int(v[0]), int(v[1]))
                else: 
                    return (int(float(v) * 10000), 10000)

            with tifffile.TiffFile(path) as tif:
                tags = tif.pages[0].tags
                if 'ColorMatrix2' in tags:
                    val = tags['ColorMatrix2'].value
                    if len(val) == 9: c_mat = [extract_exact_rational(v) for v in val]
                if c_mat is None and 'ColorMatrix1' in tags:
                    val = tags['ColorMatrix1'].value
                    if len(val) == 9: c_mat = [extract_exact_rational(v) for v in val]
                    
                if 'AsShotNeutral' in tags:
                    val = tags['AsShotNeutral'].value
                    if len(val) == 3: as_shot_neutral = [extract_exact_rational(v) for v in val]
        except Exception:
            pass 

        with rawpy.imread(path) as raw:
            bayer = raw.raw_image_visible.astype(np.float32)
            
            if as_shot_neutral is None:
                wb = raw.camera_whitebalance
                nr = int(10000 / wb[0]) if (wb is not None and len(wb) > 0 and wb[0] > 0) else 10000
                ng = int(10000 / wb[1]) if (wb is not None and len(wb) > 1 and wb[1] > 0) else 10000
                nb = int(10000 / wb[2]) if (wb is not None and len(wb) > 2 and wb[2] > 0) else 10000
                as_shot_neutral = [(nr, 10000), (ng, 10000), (nb, 10000)]

            if c_mat is None:
                try:
                    xyz_cam = raw.rgb_xyz_matrix[:3, :3]
                    if np.sum(np.abs(xyz_cam)) < 0.1: 
                        xyz_cam = np.array([[ 3.2404542, -1.5371385, -0.4985314],
                                            [-0.9692660,  1.8760108,  0.0415560],
                                            [ 0.0556434, -0.2040259,  1.0572252]])
                    # Allowing the natural negative values from the default matrix
                    c_mat = [(int(val * 10000), 10000) for val in xyz_cam.flatten()]
                except:
                    c_mat = [(10000, 10000), (0, 10000), (0, 10000), 
                             (0, 10000), (10000, 10000), (0, 10000), 
                             (0, 10000), (0, 10000), (10000, 10000)]

            # --> DETECT LINEAR DNGs
            if len(bayer.shape) == 3:
                print(f"[*] Detected Linear DNG. Routing to RGB pipeline...")
                try:
                    wl_data = raw.camera_white_level_per_channel
                    white_level = float(max(wl_data) if isinstance(wl_data, (list, tuple)) else wl_data)
                except:
                    white_level = float(np.max(bayer) or 65535.0)

                try:
                    black_levels = [int(x) for x in raw.black_level_per_channel]
                except:
                    black_levels = [0, 0, 0]

                global CAMERA_METADATA
                CAMERA_METADATA = {
                    "pattern": [0,1,1,2],
                    "black": black_levels,
                    "white": max(1, int(white_level)),
                    "as_shot_neutral": as_shot_neutral,
                    "color_matrix": c_mat
                }

                img = bayer / white_level
                return img, False 

            # --> Standard CFA (Bayer) DNG
            try:
                white_level = float(max(raw.camera_white_level_per_channel))
            except:
                white_level = float(np.max(bayer))
                
            bayer = bayer / white_level
            
            h, w = bayer.shape
            cfa_packed = np.zeros((h//2, w//2, 4), dtype=np.float32)
            cfa_packed[:, :, 0] = bayer[0::2, 0::2]
            cfa_packed[:, :, 1] = bayer[0::2, 1::2]
            cfa_packed[:, :, 2] = bayer[1::2, 0::2]
            cfa_packed[:, :, 3] = bayer[1::2, 1::2]
            
            pattern_raw = raw.raw_pattern.flatten().tolist()
            pattern_dng = [1 if x == 3 else x for x in pattern_raw]
            
            try:
                black_levels = [int(x) for x in raw.black_level_per_channel]
            except:
                black_levels = [0, 0, 0]
                
            CAMERA_METADATA = {
                "pattern": pattern_dng,
                "black": black_levels,
                "white": max(1, int(white_level)),
                "as_shot_neutral": as_shot_neutral,
                "color_matrix": c_mat
            }
            return cfa_packed, True 

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
            
        white_level = CAMERA_METADATA.get("white", 65535)
        return (np.clip(final_rgb, 0, 1) * white_level).astype(np.uint16)

# =================================================================
# 5. AUTO-TUNER
# =================================================================
def chunked_composite_loss(ref_img, recon_img, chunk_size=4096):
    h, w = ref_img.shape[:2]
    
    # 1. Normalize ref_img to [0.0, 1.0] float32
    if ref_img.dtype == np.uint8:
        ref_img = ref_img.astype(np.float32) / 255.0
    elif ref_img.dtype == np.uint16:
        ref_img = ref_img.astype(np.float32) / 65535.0
    elif ref_img.dtype in [np.float32, np.float64]:
        max_ref = float(np.max(ref_img))
        if max_ref > 1.0:
            ref_img = ref_img / max_ref

    # 2. Normalize recon_img to [0.0, 1.0] float32 independently
    if recon_img.dtype == np.uint8:
        recon_img = recon_img.astype(np.float32) / 255.0
    elif recon_img.dtype == np.uint16:
        recon_img = recon_img.astype(np.float32) / 65535.0
    elif recon_img.dtype in [np.float32, np.float64]:
        max_recon = float(np.max(recon_img))
        if max_recon > 1.0:
            recon_img = recon_img / max_recon

    total_mse, total_psnr, total_ssim, total_corr = 0, 0, 0, 0
    chunks = 0
    
    for y in range(0, h, chunk_size):
        for x in range(0, w, chunk_size):
            ref_chunk = ref_img[y:y+chunk_size, x:x+chunk_size]
            recon_chunk = recon_img[y:y+chunk_size, x:x+chunk_size]
            
            if ref_chunk.shape[0] < 7 or ref_chunk.shape[1] < 7:
                continue
                
            mse = np.mean((ref_chunk - recon_chunk) ** 2)
            psnr = 10 * np.log10(1.0 / mse) if mse > 1e-10 else 100.0
            corr = correlation(ref_chunk.flatten(), recon_chunk.flatten())
            
            is_multi = len(ref_chunk.shape) == 3
            s_val = ssim(ref_chunk, recon_chunk, channel_axis=2 if is_multi else None, data_range=1.0)
            
            total_mse += mse
            total_psnr += psnr
            total_corr += corr
            total_ssim += s_val
            chunks += 1
            
    if chunks == 0:
        return float('inf'), float('inf'), 0.0, 0.0, 0.0
        
    avg_mse = total_mse / chunks
    avg_psnr = total_psnr / chunks
    avg_corr = total_corr / chunks
    avg_ssim = total_ssim / chunks
    
    norm_mse = min(avg_mse * 10, 1.0)
    norm_psnr = max(0.0, (50.0 - avg_psnr) / 50.0) 
    norm_ssim = 1.0 - avg_ssim
    
    composite_loss = (0.4 * norm_ssim) + (0.3 * norm_psnr) + (0.2 * (1.0 - avg_corr)) + (0.1 * norm_mse)
    
    return composite_loss, avg_mse, avg_psnr, avg_ssim, avg_corr

def run_auto_tuner(folder_path, config, valid_exts, use_full=True):
    print(f"[*] Initializing Full-Reference Auto-Tuner...")
    print(f"[*] Hardware limits enforced: {config.get('MAX_THREADS', 'All')} Cores, {config.get('MAX_RAM_GB', 30)} GB RAM")
    
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]
    if not files:
        print("[!] No valid images found for tuning.")
        return config

    base_l = config.get("LAMBDA_MIN", 1e-4)
    base_t = config.get("TV_WEIGHT", 0.005)
    scale = config.get("SCALE_FACTOR", 2)
    
    # 3x3 Grid Construction
    l_grid = np.logspace(np.log10(base_l) - 1, np.log10(base_l) + 1, 3).tolist()
    t_grid = np.linspace(base_t * 0.5, base_t * 1.5, 3).tolist()
    scores = {(l, t): [] for l in l_grid for t in t_grid}

    print(f"[*] Testing {len(l_grid) * len(t_grid)} parameter combinations on {len(files)} files...")
    
    for f in files:
        if not KEEP_RUNNING: break
        
        gt_img, is_cfa = load_image(f)
        if gt_img is None: continue
        
        print(f"\n[*] Tuning on {os.path.basename(f)}...")
        h, w = gt_img.shape[:2]

        # 1. Simulate Sensor Data (Degrade the image using Area Interpolation to prevent moiré)
        new_w, new_h = w // scale, h // scale
        simulated_sensor_data = cv2.resize(gt_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Lock the ground truth dimensions so they perfectly match the reconstruction output
        gt_img_cropped = gt_img[:new_h * scale, :new_w * scale]
        
        for l_val in l_grid:
            for t_val in t_grid:
                if not KEEP_RUNNING: break

                step_start_time = time.time()

                # 2. Reconstruct (Hardware limits are enforced internally by process_full)
                recon_img = process_full(simulated_sensor_data, is_cfa, config, l_override=l_val, t_override=t_val)
                
                # 3. Memory-Safe Chunked Evaluation
                # FIX: Unpack the new return values
                loss, mse_val, psnr_val, ssim_val, corr_val = chunked_composite_loss(gt_img_cropped, recon_img, chunk_size=4096)
                scores[(l_val, t_val)].append(loss)

                step_duration = time.time() - step_start_time
                # Log to the CSV immediately so it can be monitored mid-tune
                logger.log_entry(
                    file_name=os.path.basename(f), 
                    scale=scale,
                    l_val=l_val, 
                    t_val=t_val, 
                    duration=step_duration, 
                    mse=mse_val,       
                    psnr=psnr_val,
                    ssim=ssim_val,
                    corr=corr_val,
                    comp_loss=loss
                )
                print(f"    -> L:{l_val:.2e}, T:{t_val:.4f} | Composite Loss: {loss:.4f} (Lower is better)")

    # Directory Averaging & Parameter Selection
    best_loss = float('inf')
    best_params = {"LAMBDA_MIN": config["LAMBDA_MIN"], "TV_WEIGHT": config["TV_WEIGHT"]}

    print("\n[*] --- Auto-Tuning Results (Directory Averages) ---")
    for (l, t), score_list in scores.items():
        if not score_list: continue
        
        avg_loss = sum(score_list) / len(score_list)
        print(f"    -> L:{l:.2e}, T:{t:.4f} | Avg Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_params["LAMBDA_MIN"] = l
            best_params["TV_WEIGHT"] = t

    config.update(best_params)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(config, f, indent=4)
        
    print(f"[*] Auto-Tuner Complete. Optimal Parameters saved to settings.json: {best_params}")
    return config

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
    parser.add_argument("--tune", help="Run Auto-Tuner on a directory", action="store_true")
    parser.add_argument("--full", action="store_true", help="Use full image for auto-tuning instead of 512x512 center crop")
    parser.add_argument("--dng", action="store_true", help="Output as DNG (True RAW)")
    
    args = parser.parse_args()

    cfg = {"SCALE_FACTOR": 2, "TILE_SIZE": 64, "OVERLAP": 16, "LAMBDA_MIN": 1e-4, "TV_WEIGHT": 0.005, "MAX_THREADS": 48, "MAX_RAM_GB": 30}
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f: cfg.update(json.load(f))
        
    valid_exts = ('.dng', '.arw', '.cr2', '.nef', '.orf', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')

    if args.tune and args.input:
        cfg = run_auto_tuner(args.input, cfg, valid_exts, use_full=args.full)
        sys.exit(0)

    if args.test:
        f = args.test
        img, is_cfa = load_image(f)
        if img is not None:
            for l_val in [1e-6, 1e-5, 1e-4]:
                for t_val in [0.001, 0.002, 0.005]:
                    if not KEEP_RUNNING: break
                    res = process_full(img, is_cfa, cfg, l_val, t_val)
                    if is_cfa or args.dng:
                        out = f"{os.path.splitext(f)[0]}_L{l_val}_T{t_val}.dng"
                        MonolithDNGWriter().save(res, out, is_cfa)
                    else:
                        out = f"{os.path.splitext(f)[0]}_L{l_val}_T{t_val}.tif"
                        # Ensure 3-channel RGB for safe TIFF writing
                        if len(res.shape) == 3 and res.shape[2] > 3:
                            res = res[:, :, :3]
                        tifffile.imwrite(
                            out, 
                            (res * 65535.0).astype(np.uint16),
                            bigtiff=True, 
                            compression='zlib', 
                            photometric='rgb'
                        )

    elif args.batch or args.input:
        path = args.batch if args.batch else args.input
        files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(valid_exts)] if os.path.isdir(path) else [path]
        for f in files:
            if not KEEP_RUNNING: break
            
            img, is_cfa = load_image(f)
            if img is not None:
                force_dng = args.dng 
                out_ext = ".dng" if (is_cfa or force_dng) else ".tif"
                out_path = f"{os.path.splitext(f)[0]}_irk_v57{out_ext}"
                
                if os.path.exists(out_path): 
                    print(f"[SKIP] {out_path} already exists. Resuming queue.")
                    continue
                
                start_time = time.time()
                res = process_full(img, is_cfa, cfg)
                
                if out_ext == ".dng":
                    MonolithDNGWriter().save(res, out_path, is_cfa=is_cfa)
                else:
                    # Safe fallback in case data wasn't converted
                    if res.dtype == np.float32:
                        res = (np.clip(res, 0, 1) * 65535.0).astype(np.uint16)
                    
                    # Ensure 3-channel RGB to prevent alpha bugs
                    if len(res.shape) == 3 and res.shape[2] > 3:
                        res = res[:, :, :3]
                        
                    tifffile.imwrite(
                        out_path, 
                        res,
                        bigtiff=True, 
                        compression='zlib', 
                        photometric='rgb'
                    )
                    
                logger.log_entry(os.path.basename(f), cfg["SCALE_FACTOR"], cfg["LAMBDA_MIN"], cfg["TV_WEIGHT"], time.time()-start_time)
                print(f"[COMPLETE] Saved: {out_path}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()