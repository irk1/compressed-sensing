import rawpy
import cv2
import numpy as np
import imageio

def repair_dng_native(input_path, output_path):
    # 1. High-Quality RAW Development
    with rawpy.imread(input_path) as raw:
        # We use AHD demosaicing and 16-bit depth to keep every bit of sensor data
        rgb = raw.postprocess(use_camera_wb=True, 
                              no_auto_bright=False, 
                              output_bps=16, 
                              demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD)

    # Convert to float32 for high-precision processing
    img_float = rgb.astype(np.float32) / 65535.0

    # --- STAGE 1: SIGNAL-BASED DENOISING ---
    # Fast Non-Local Means is much better at "repairing" noise than simple filters
    # because it searches for similar pixel patterns across the frame.
    img_8bit = (img_float * 255).astype(np.uint8)
    denoised_8bit = cv2.fastNlMeansDenoisingColored(img_8bit, None, h=10, hColor=10, 
                                                    templateWindowSize=7, searchWindowSize=21)
    denoised = denoised_8bit.astype(np.float32) / 255.0

    # --- STAGE 2: MULTI-SCALE SHARPENING (The "Topaz" Look) ---
    # We create two layers of sharpening: Fine Detail and Medium Detail.
    
    # Layer A: Fine Detail (Micro-contrast)
    blur_fine = cv2.GaussianBlur(denoised, (0, 0), 0.5)
    fine_detail = cv2.addWeighted(denoised, 1.5, blur_fine, -0.5, 0)

    # Layer B: Medium Detail (Structural Sharpening)
    blur_med = cv2.GaussianBlur(denoised, (0, 0), 2.0)
    med_detail = cv2.addWeighted(denoised, 1.2, blur_med, -0.2, 0)

    # Combine them using an Edge Mask so we don't sharpen "flat" areas
    gray = cv2.cvtColor(denoised, cv2.COLOR_RGB2GRAY)
    mask = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    mask = cv2.GaussianBlur(np.abs(mask), (5, 5), 0)
    mask = np.clip(mask * 15, 0, 1) # Sensitivity adjustment

    # Blend the layers: Background (Denoised) -> Medium Edges -> Fine Edges
    sharpened_stack = (med_detail * 0.4) + (fine_detail * 0.6)
    final = (denoised * (1 - mask[:,:,None]) + sharpened_stack * mask[:,:,None])

    # --- STAGE 3: LOCAL CONTRAST BOOST ---
    # This emulates the 'Clarity' and 'Texture' sliders.
    final_8bit = (np.clip(final, 0, 1) * 255).astype(np.uint8)
    enhanced_8bit = cv2.detailEnhance(final_8bit, sigma_s=10, sigma_r=0.15)
    
    # Convert back to 16-bit for the final high-quality export
    final_16bit = (enhanced_8bit.astype(np.float32) / 255.0 * 65535).astype(np.uint16)

    imageio.imwrite(output_path, final_16bit)
    print(f"Repair complete: {output_path}")

# Run on your server
repair_dng_native('input.dng', 'repaired_pro.tiff')