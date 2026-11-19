import os
import sys
from datetime import datetime
import cv2
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import correlation


def load_high_bitdepth(path):
    """Loads 8-bit, 16-bit, or 32-bit float RGB images without bit-depth truncation."""
    ext = os.path.splitext(path)[1].lower()
    if ext in [".tif", ".tiff"]:
        img = tifffile.imread(path)
    elif ext == ".dng":
        import rawpy
        with rawpy.imread(path) as raw:
            img = raw.postprocess(output_bps=16, no_auto_bright=True, use_camera_wb=True)
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None and img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]

    bit_depth = img.dtype
    if img.dtype == np.uint8:
        arr = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        arr = img.astype(np.float32) / 65535.0
    else:
        arr = np.clip(img.astype(np.float32), 0.0, 1.0)

    return arr, str(bit_depth)


def compute_gmsd(ref, dist):
    """Gradient Magnitude Similarity Deviation (lower is better; catches blur & ringing)."""
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
    dist_gray = cv2.cvtColor(dist, cv2.COLOR_RGB2GRAY)

    # Prewitt gradient filters
    kx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32) / 3.0
    ky = kx.T

    g_ref = np.sqrt(cv2.filter2D(ref_gray, -1, kx) ** 2 + cv2.filter2D(ref_gray, -1, ky) ** 2)
    g_dist = np.sqrt(cv2.filter2D(dist_gray, -1, kx) ** 2 + cv2.filter2D(dist_gray, -1, ky) ** 2)

    c = 0.0026  # Standard GMSD stability constant for [0, 1] range
    gms = (2.0 * g_ref * g_dist + c) / (g_ref ** 2 + g_dist ** 2 + c)
    return float(np.std(gms))


def compute_tile_seam_ratio(img, stride=96):
    """Measures boundary gradient discontinuity at tile strides vs interior pixels."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    grad_y = np.abs(np.diff(gray, axis=0))
    grad_x = np.abs(np.diff(gray, axis=1))

    seam_rows = np.arange(stride - 1, grad_y.shape[0], stride)
    seam_cols = np.arange(stride - 1, grad_x.shape[1], stride)

    if len(seam_rows) == 0 or len(seam_cols) == 0:
        return 1.0

    seam_energy = 0.5 * (np.mean(grad_y[seam_rows, :]) + np.mean(grad_x[:, seam_cols]))
    global_energy = 0.5 * (np.mean(grad_y) + np.mean(grad_x)) + 1e-8
    return float(seam_energy / global_energy)


def generate_pdf_report(orig_path, recon_path, heatmap_ceiling=0.02, tile_stride=96):
    try:
        ref_raw, ref_dtype = load_high_bitdepth(orig_path)
        recon_raw, recon_dtype = load_high_bitdepth(recon_path)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    h_ref, w_ref = ref_raw.shape[:2]
    h_rec, w_rec = recon_raw.shape[:2]

    # Auto-detect Evaluation Mode
    if (h_ref, w_ref) == (h_rec, w_rec):
        mode = "Full-Resolution Ground-Truth Benchmark"
        ref = ref_raw
        recon = recon_raw
        # Generate Lanczos baseline from 2x downsampled reference to measure true CS gain
        lr_sim = cv2.resize(ref, (w_ref // 2, h_ref // 2), interpolation=cv2.INTER_AREA)
        baseline = cv2.resize(lr_sim, (w_ref, h_ref), interpolation=cv2.INTER_LANCZOS4)
        base_psnr = 10 * np.log10(1.0 / max(np.mean((ref - baseline) ** 2), 1e-12))
        base_ssim = ssim(ref, baseline, channel_axis=2, data_range=1.0)
    else:
        mode = "Downsampled Data-Consistency Check (||Ax - y||)"
        ref = ref_raw
        # Use INTER_AREA (box-averaging) to match physical sensor binning
        recon = cv2.resize(recon_raw, (w_ref, h_ref), interpolation=cv2.INTER_AREA)
        base_psnr, base_ssim = None, None

    # Core Fidelity Metrics
    mse = float(np.mean((ref - recon) ** 2))
    psnr = float(10 * np.log10(1.0 / mse)) if mse > 0 else 100.0
    pearson_r = float(1.0 - correlation(ref.flatten(), recon.flatten()))
    ssim_val = float(ssim(ref, recon, channel_axis=2, data_range=1.0))
    gmsd_val = compute_gmsd(ref, recon)
    seam_ratio = compute_tile_seam_ratio(recon_raw, stride=tile_stride)

    # Fixed-Scale Error Heatmap (Anchored to heatmap_ceiling for objective run-to-run comparison)
    diff_gray = np.mean(np.abs(ref - recon), axis=2)
    diff_clipped = np.clip(diff_gray / heatmap_ceiling, 0.0, 1.0)

    # Create PDF Report
    pdf_path = "Comparison_Report.pdf"
    fig = plt.figure(figsize=(13, 16))

    plt.suptitle("IRK_CS-Upscale Reconstruction Fidelity Report", fontsize=20, color="#003366", fontweight="bold")
    subtitle = (
        f"Mode: {mode}  |  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Ref: {w_ref}x{h_ref} ({ref_dtype})  vs.  Recon: {w_rec}x{h_rec} ({recon_dtype})"
    )
    plt.figtext(0.5, 0.915, subtitle, ha="center", fontsize=10)

    # Top Row Visuals
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.imshow(np.clip(ref, 0, 1))
    ax1.set_title("Reference Image")
    ax1.axis("off")

    ax2 = fig.add_subplot(3, 3, 2)
    ax2.imshow(np.clip(recon, 0, 1))
    ax2.set_title("Reconstructed (Aligned)")
    ax2.axis("off")

    ax3 = fig.add_subplot(3, 3, 3)
    im = ax3.imshow(diff_clipped, cmap="jet", vmin=0.0, vmax=1.0)
    ax3.set_title(f"Fixed-Scale Error (Max = {heatmap_ceiling*100:.1f}%)")
    ax3.axis("off")
    cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["0%", f"{heatmap_ceiling*50:.1f}%", f"≥{heatmap_ceiling*100:.1f}%"])

    # Metrics Table
    psnr_note = f"Target: >30 dB (Lanczos: {base_psnr:.2f} dB)" if base_psnr else "Target: >35 dB (Data Consistency)"
    ssim_note = f"Target: >0.980 (Lanczos: {base_ssim:.4f})" if base_ssim else "Target: >0.990 (Data Consistency)"

    columns = ("Metric", "Value", "Benchmark / Target")
    data = [
        ["MSE (16-bit Float)", f"{mse:.7f}", "Lower is better (Ideal: 0.0)"],
        ["PSNR", f"{psnr:.2f} dB", psnr_note],
        ["SSIM", f"{ssim_val:.5f}", ssim_note],
        ["Pearson Correlation (r)", f"{pearson_r:.5f}", "Higher is better (Target: >0.990)"],
        ["GMSD (Edge/Ringing)", f"{gmsd_val:.5f}", "Lower is better (Target: <0.030)"],
        ["Tile-Seam Ratio", f"{seam_ratio:.3f}x", "Ideal: ~1.00x (>1.15x = grid seams)"]
    ]

    table_ax = fig.add_subplot(3, 1, 2)
    table_ax.axis("tight")
    table_ax.axis("off")
    the_table = table_ax.table(cellText=data, colLabels=columns, loc="center", cellLoc="center")
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(11)
    the_table.scale(1.15, 2.2)

    explanation = (
        "Diagnostic Notes:\n"
        "• Full-Res Ground-Truth Mode triggers automatically when Reference and Recon have identical dimensions.\n"
        "• GMSD measures gradient distortion—spikes when TV_WEIGHT is too high (blur) or too low (ECOS ringing).\n"
        "• Tile-Seam Ratio compares edge energy at tile boundaries against image interior (flags OVERLAP issues).\n"
        f"• Heatmap is locked to a {heatmap_ceiling*100:.1f}% full-scale error ceiling so PDFs can be compared across runs."
    )
    plt.figtext(0.08, 0.08, explanation, fontsize=10.5, ha="left", bbox=dict(facecolor="none", edgecolor="gray", pad=10.0))

    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Success! Report saved to: {os.path.abspath(pdf_path)}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python cs_verify.py <Original_or_GroundTruth_Path> <Recon_Path>")
    else:
        generate_pdf_report(sys.argv[1], sys.argv[2])