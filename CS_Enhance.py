import sys
import cv2
import numpy as np
from scipy.ndimage import convolve, gaussian_filter, median_filter
from skimage import restoration
from skimage.restoration import denoise_nl_means

try:
    import bm3d
    HAS_BM3D = True
except ImportError:
    HAS_BM3D = False


def _ensure_float32(image):
    """Guarantees [0.0, 1.0] range and C-contiguous float32 memory layout."""
    return np.ascontiguousarray(np.clip(image, 0.0, 1.0), dtype=np.float32)


def _apply_per_channel(image, fn):
    """Applies a 2D single-channel function independently across 2D, 3-ch RGB, or 4-ch CFA."""
    img = _ensure_float32(image)
    if img.ndim == 2:
        return _ensure_float32(fn(img))
    out = np.empty_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        out[:, :, c] = fn(img[:, :, c])
    return _ensure_float32(out)


def generate_psf(size=5, sigma=1.0):
    """Generates a normalized 2D Gaussian Point Spread Function."""
    radius = size // 2
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    g = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * (sigma ** 2)))
    return (g / np.sum(g)).astype(np.float32)


def estimate_noise_sigma(image):
    """Immerkaer Laplacian MAD Gaussian noise estimator (pure NumPy/SciPy)."""
    img = _ensure_float32(image)
    kernel = np.array(
        [[1.0, -2.0, 1.0],
         [-2.0, 4.0, -2.0],
         [1.0, -2.0, 1.0]],
        dtype=np.float32,
    )
    norm_factor = 6.0 * 0.67448975

    def _ch_sigma(ch):
        if ch.shape[0] < 5 or ch.shape[1] < 5:
            return 0.0
        resp = convolve(ch, kernel, mode="reflect")[1:-1, 1:-1]
        return float(np.median(np.abs(resp)) / norm_factor)

    if img.ndim == 3:
        return float(np.median([_ch_sigma(img[:, :, c]) for c in range(img.shape[2])]))
    return _ch_sigma(img)


def anscombe_forward(image, gain=4095.0):
    """Forward Anscombe VST: stabilizes Poisson photon shot noise on linear RAWs."""
    z = np.maximum(_ensure_float32(image) * float(gain), 0.0)
    y = 2.0 * np.sqrt(z + 0.375)
    max_y = 2.0 * np.sqrt(float(gain) + 0.375)
    return np.ascontiguousarray(y / max_y, dtype=np.float32), float(max_y)


def anscombe_inverse(y_norm, max_y, gain=4095.0):
    """Exact unbiased inverse Anscombe transform (Makitalo & Foi)."""
    y = np.maximum(np.asarray(y_norm, dtype=np.float32) * float(max_y), 1.2247449)
    inv_y = 1.0 / y
    z_hat = (
        0.25 * (y ** 2)
        - 0.125
        + 0.25 * np.sqrt(1.5) * inv_y
        - 1.375 * (inv_y ** 2)
        + 0.625 * np.sqrt(1.5) * (inv_y ** 3)
    )
    return _ensure_float32(z_hat / float(gain))


def remove_hot_pixels(image, threshold=0.20):
    """Suppresses isolated hot/stuck sensor pixels via 3x3 per-channel median check."""
    def _clean(ch):
        med = median_filter(ch, size=3)
        return np.where(np.abs(ch - med) > float(threshold), med, ch)

    return _apply_per_channel(image, _clean)


def apply_bm3d(image, noise_std_dev=None, fast_mode=False, use_anscombe=False, sensor_gain=4095.0):
    """
    Block-Matching 3D (BM3D) collaborative filtering.
    - Supports 2D grayscale, 3-ch RGB (joint volume), and 4-ch Bayer CFA (isolated planes).
    - Pass noise_std_dev=None to auto-estimate the sensor noise floor.
    """
    if not HAS_BM3D:
        print("[WARN] bm3d library not found; falling back to Non-Local Means.")
        return apply_nlm(image, noise_std_dev=noise_std_dev)

    img = _ensure_float32(image)
    stage = (
        bm3d.BM3DStages.HARD_THRESHOLDING
        if fast_mode
        else bm3d.BM3DStages.ALL_STAGES
    )

    if use_anscombe:
        work_img, max_y = anscombe_forward(img, gain=sensor_gain)
        eff_sigma = float(noise_std_dev) if noise_std_dev is not None else (1.0 / max_y)
    else:
        work_img = img
        eff_sigma = float(noise_std_dev) if noise_std_dev is not None else estimate_noise_sigma(work_img)

    if eff_sigma <= 1e-6:
        return img

    if work_img.ndim == 3:
        if work_img.shape[2] == 3:
            denoised = bm3d.bm3d(work_img, sigma_psd=eff_sigma, stage_arg=stage)
        else:
            denoised = np.empty_like(work_img, dtype=np.float32)
            for c in range(work_img.shape[2]):
                denoised[:, :, c] = bm3d.bm3d(
                    work_img[:, :, c], sigma_psd=eff_sigma, stage_arg=stage
                )
    else:
        denoised = bm3d.bm3d(work_img, sigma_psd=eff_sigma, stage_arg=stage)

    if use_anscombe:
        return anscombe_inverse(denoised, max_y, gain=sensor_gain)
    return _ensure_float32(denoised)


def apply_nlm(image, noise_std_dev=None, patch_size=5, patch_distance=6):
    """Non-Local Means (NLM) patch-similarity denoising."""
    s = float(noise_std_dev) if noise_std_dev is not None else estimate_noise_sigma(image)
    if s <= 1e-6:
        return _ensure_float32(image)

    return _apply_per_channel(
        image,
        lambda ch: denoise_nl_means(
            ch,
            h=1.15 * s,
            fast_mode=True,
            patch_size=int(patch_size),
            patch_distance=int(patch_distance),
            channel_axis=None,
        ),
    )


def apply_bilateral(image, sigma_color=0.03, sigma_spatial=2.0):
    """Fast edge-preserving bilateral filter."""
    return _apply_per_channel(
        image,
        lambda ch: cv2.bilateralFilter(
            ch, d=-1, sigmaColor=float(sigma_color), sigmaSpace=float(sigma_spatial)
        ),
    )


def apply_richardson_lucy(image, iterations=15, psf_size=5, psf_sigma=1.2):
    """Iterative Richardson-Lucy deconvolution (2D, 3-ch RGB, or 4-ch CFA)."""
    if iterations <= 0:
        return _ensure_float32(image)

    psf = generate_psf(size=int(psf_size), sigma=float(psf_sigma))
    img_safe = np.clip(_ensure_float32(image), 1e-6, 1.0)

    return _apply_per_channel(
        img_safe,
        lambda ch: restoration.richardson_lucy(
            ch, psf, num_iter=int(iterations), clip=False
        ),
    )


def apply_unsharp_mask(image, radius=1.5, amount=0.6):
    """High-frequency unsharp mask sharpening."""
    if amount <= 0:
        return _ensure_float32(image)

    def _usm(ch):
        blurred = gaussian_filter(ch, sigma=float(radius))
        return ch + float(amount) * (ch - blurred)

    return _apply_per_channel(image, _usm)


def apply_local_contrast(image, clip_limit=2.0, tile_grid=8):
    """16-bit luminance-locked CLAHE local contrast enhancement."""
    img = _ensure_float32(image)
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit), tileGridSize=(int(tile_grid), int(tile_grid))
    )

    if img.ndim == 3 and img.shape[2] == 3:
        lum = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    elif img.ndim == 3:
        lum = np.mean(img, axis=2)
    else:
        lum = img

    lum_u16 = np.round(np.clip(lum, 0.0, 1.0) * 65535.0).astype(np.uint16)
    lum_eq = clahe.apply(lum_u16).astype(np.float32) / 65535.0
    gain_map = (lum_eq + 1e-5) / (lum + 1e-5)

    if img.ndim == 3:
        return _ensure_float32(img * gain_map[:, :, None])
    return _ensure_float32(lum_eq)


def run_pre_cs_pipeline(image, is_cfa=False, settings=None, sensor_white=4095.0):
    """Pre-solver enhancement hook triggered by `--enhance`."""
    cfg = settings or {}
    out = _ensure_float32(image)

    if cfg.get("HOT_PIXEL_CLEAN", True):
        out = remove_hot_pixels(out, threshold=float(cfg.get("HOT_PIXEL_THRESH", 0.20)))

    method = str(cfg.get("ENHANCE_METHOD", "bm3d")).lower()
    sigma = cfg.get("DENOISE_SIGMA", None)

    if method == "nlm":
        out = apply_nlm(out, noise_std_dev=sigma)
    elif method == "bilateral":
        out = apply_bilateral(out)
    else:
        out = apply_bm3d(
            out,
            noise_std_dev=sigma,
            fast_mode=bool(cfg.get("DENOISE_FAST", False)),
            use_anscombe=bool(cfg.get("USE_ANSCOMBE", is_cfa)),
            sensor_gain=float(sensor_white),
        )

    return _ensure_float32(out)


def run_post_cs_pipeline(image, is_cfa=False, settings=None):
    """Post-solver enhancement hook (ready for later post-CS deconvolution/sharpening)."""
    cfg = settings or {}
    out = _ensure_float32(image)

    rl_iters = int(cfg.get("SHARPEN_ITERS", 0))
    if rl_iters > 0:
        default_sigma = 0.75 if is_cfa else 1.2
        out = apply_richardson_lucy(
            out,
            iterations=rl_iters,
            psf_size=int(cfg.get("PSF_SIZE", 5)),
            psf_sigma=float(cfg.get("PSF_SIGMA", default_sigma)),
        )

    usm_amount = float(cfg.get("USM_AMOUNT", 0.0))
    if usm_amount > 0.0:
        out = apply_unsharp_mask(
            out, radius=float(cfg.get("USM_RADIUS", 1.5)), amount=usm_amount
        )

    if cfg.get("LOCAL_CONTRAST", False):
        out = apply_local_contrast(out, clip_limit=float(cfg.get("CLAHE_CLIP", 2.0)))

    return _ensure_float32(out)