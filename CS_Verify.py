import cv2
import numpy as np
import sys
import os
from skimage.metrics import structural_similarity as ssim

def verify_reconstruction(original_path, recon_path):
    if not os.path.exists(original_path) or not os.path.exists(recon_path):
        print("[!] Error: One or both files missing.")
        return

    # Load images
    orig = cv2.imread(original_path)
    recon = cv2.imread(recon_path)

    # 1. Align Dimensions
    # If the reconstruction is 2x the original, we upsample the original 
    # using a high-quality Lanczos filter to provide a fair baseline.
    h_r, w_r = recon.shape[:2]
    h_o, w_o = orig.shape[:2]

    if (h_r, w_r) != (h_o, w_o):
        print(f"[*] Resizing original ({w_o}x{h_o}) to match reconstruction ({w_r}x{h_r})...")
        orig_scaled = cv2.resize(orig, (w_r, h_r), interpolation=cv2.INTER_LANCZOS4)
    else:
        orig_scaled = orig

    # 2. Calculate Mathematical Accuracy (PSNR)
    mse = np.mean((orig_scaled.astype(np.float32) - recon.astype(np.float32))**2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100

    # 3. Calculate Structural Similarity (SSIM)
    # This measures contrast, luminance, and texture preservation
    score, diff = ssim(orig_scaled, recon, channel_axis=2, full=True)
    
    # 4. Generate Error Heatmap
    # We take the absolute difference and amplify it so errors are visible
    error_map = cv2.absdiff(orig_scaled, recon)
    error_map = cv2.applyColorMap(cv2.cvtColor(error_map, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)

    # 5. Output Results
    print(f"\n" + "="*40)
    print(f"VERIFICATION REPORT")
    print(f"="*40)
    print(f"PSNR (Higher is better): {psnr:.2f} dB")
    print(f"SSIM (Closer to 1.0 is better): {score:.4f}")
    
    # Verdict
    if score > 0.95: verdict = "EXCELLENT (Production Grade)"
    elif score > 0.85: verdict = "GOOD (Minor artifacts)"
    else: verdict = "POOR (Tuning required)"
    print(f"VERDICT: {verdict}")
    print(f"="*40)

    # Save the visual evidence
    cv2.imwrite("VERIFICATION_HEATMAP.png", error_map)
    print("[*] Saved 'VERIFICATION_HEATMAP.png'. Blue = Match, Red = Error.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 CS_Verify.py <Original_HighRes> <Reconstructed_HighRes>")
    else:
        verify_reconstruction(sys.argv[1], sys.argv[2])