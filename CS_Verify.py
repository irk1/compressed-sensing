import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import correlation
import sys
import os
from datetime import datetime

def generate_pdf_report(orig_path, recon_path):
    # 1. Load Images
    img1 = cv2.imread(orig_path)
    img2 = cv2.imread(recon_path)
    
    if img1 is None or img2 is None:
        print("Error: Files not found. Check your paths.")
        return

    # Convert BGR (OpenCV default) to RGB for Matplotlib
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # 2. Standardize Dimensions (Matches Mathematica Resize)
    h, w = img1_rgb.shape[:2]
    img2_res = cv2.resize(img2_rgb, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # Convert to float [0,1] for high-precision math
    ref = img1_rgb.astype(np.float32) / 255.0
    recon = img2_res.astype(np.float32) / 255.0

    # 3. Statistical Calculations
    mse = np.mean((ref - recon) ** 2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 100
    corr = correlation(ref.flatten(), recon.flatten())
    # SSIM (Calculated on 2048px for stability if images are massive)
    ssim_val = ssim(ref, recon, channel_axis=2, data_range=1.0)

    # 4. Heatmap Generation
    diff = np.abs(ref - recon)
    diff_gray = np.mean(diff, axis=2)
    # Rescale error to show full color spectrum
    diff_norm = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    # Add tight 2-pixel black border
    heatmap_bordered = cv2.copyMakeBorder(heatmap_rgb, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # 5. Create PDF Layout using Matplotlib
    pdf_path = "Comparison_Report.pdf"
    fig = plt.figure(figsize=(12, 16))
    
    # Title and Date
    plt.suptitle("Image Comparison & Reconstruction Report", fontsize=22, color='#003366', fontweight='bold')
    plt.figtext(0.5, 0.92, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ha="center", fontsize=10)

    # Image Grid (Top Row)
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.imshow(img1_rgb)
    ax1.set_title("Original Reference")
    ax1.axis('off')

    ax2 = fig.add_subplot(3, 3, 2)
    ax2.imshow(img2_res)
    ax2.set_title("Reconstructed Image")
    ax2.axis('off')

    ax3 = fig.add_subplot(3, 3, 3)
    ax3.imshow(heatmap_bordered)
    ax3.set_title("Error Heatmap")
    ax3.axis('off')

    # Metrics Table
    columns = ("Metric", "Value", "Interpretation")
    data = [
        ["MSE", f"{mse:.6f}", "Lower is better (Ideal: 0)"],
        ["PSNR", f"{psnr:.2f} dB", "Higher is better (Ideal: >30 dB)"],
        ["Correlation", f"{corr:.4f}", "Lower is better (Ideal: 0)"],
        ["SSIM", f"{ssim_val:.4f}", "Higher is better (Ideal: 1.0)"]
    ]
    
    table_ax = fig.add_subplot(3, 1, 2)
    table_ax.axis('tight')
    table_ax.axis('off')
    the_table = table_ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)
    the_table.scale(1.2, 2.5)

    # Glossary / Explanation text
    explanation = (
        "Understanding the Metrics:\n"
        "• MSE: Average squared intensity difference.\n"
        "• PSNR: Signal strength vs noise ratio (Log scale).\n"
        "• Correlation: Structural pattern divergence.\n"
        "• SSIM: Human-perceived quality based on luminance, contrast, and structure.\n\n"
        "Heatmap Key: Deep Blue = Identical | Bright Red = Maximum Error."
    )
    plt.figtext(0.1, 0.1, explanation, fontsize=11, ha="left", bbox=dict(facecolor='none', edgecolor='gray', pad=10.0))

    # Save to PDF
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"Success! Report saved to: {os.path.abspath(pdf_path)}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <Original_Path> <Recon_Path>")
    else:
        generate_pdf_report(sys.argv[1], sys.argv[2])