import numpy as np
import cv2
import cvxpy as cp
import json
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

# --- SETTINGS MANAGEMENT ---
SETTINGS_FILE = "settings.json"

DEFAULT_SETTINGS = {
    "SCALE_FACTOR": 2,
    "TRIALS": 10,
    "FINAL_ITERS": 4000,
    "TEST_SIZE": 32,
    "LAMBDA_MIN": 1e-4,
    "LAMBDA_MAX": 1e-1,
    "TV_WEIGHT": 0.01,    # NEW: Forces the 'grid' to blend
    "VISUALIZE": True,
    "BEST_PSNR_FOUND": 0.0 
}

def load_settings():
    if not os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(DEFAULT_SETTINGS, f, indent=4)
        return DEFAULT_SETTINGS
    with open(SETTINGS_FILE, 'r') as f:
        return json.load(f)

def save_settings(config):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"\n[!] SAVING: New Record {config['BEST_PSNR_FOUND']:.2f} dB.")

# --- 1. MATH ENGINE ---
def get_dct_matrix(n):
    d = np.zeros((n, n))
    for k in range(n):
        for i in range(n):
            if k == 0: d[k, i] = np.sqrt(1/n)
            else: d[k, i] = np.sqrt(2/n) * np.cos(np.pi * k * (2*i + 1) / (2*n))
    return d

# --- 2. DYNAMIC MACRO GENERATOR ---
def generate_macro_test_image(size=32):
    img = np.zeros((size, size), dtype=np.float32)
    Y, X = np.ogrid[:size, :size]
    cx, cy = np.random.randint(size//4, 3*size//4, size=2)
    img = 150 * np.exp(-np.sqrt((X-cx)**2 + (Y-cy)**2) / (size/4)) 
    freq = np.random.uniform(0.5, 1.5)
    veins = np.sin(X * freq + np.random.uniform(0, 6.28)) * 30 
    x1, y1 = np.random.randint(0, size//2, size=2)
    x2, y2 = np.random.randint(size//2, size, size=2)
    cv2.line(img, (x1, y1), (x2, y2), 255, 1)
    return np.clip(img + veins, 20, 255)

# --- 3. THE TV-ENHANCED SOLVER ---
def enlarge_with_tv_cs(low_res, ground_truth, config):
    scale = config["SCALE_FACTOR"]
    h, w = low_res.shape
    high_h, high_w = h * scale, w * scale
    D_h, D_w = get_dct_matrix(high_h), get_dct_matrix(high_w)
    
    best_img = None
    improved = False
    
    print(f"--- Running {config['TRIALS']} TV-Hybrid Trials ---")

    for i in range(config["TRIALS"]):
        test_lam = np.exp(np.random.uniform(np.log(config["LAMBDA_MIN"]), 
                                            np.log(config["LAMBDA_MAX"])))
        
        # New Tuning Variable: TV Weight
        # We allow the program to 'learn' the best smoothness too
        test_tv = config["TV_WEIGHT"] * np.random.uniform(0.5, 2.0)
        
        X_dct = cp.Variable((high_h, high_w))
        # Reconstruct spatial image to apply TV
        reconstructed_img = D_h.T @ X_dct @ D_w
        
        constraints = [reconstructed_img[::scale, ::scale] == low_res]
        
        # OBJECTIVE: Sparsity + Total Variation (Smoothness)
        # cp.tv(reconstructed_img) minimizes the difference between adjacent pixels
        sparsity_loss = cp.norm(X_dct, 1) * test_lam
        smoothness_loss = cp.tv(reconstructed_img) * test_tv
        
        obj = cp.Minimize(sparsity_loss + smoothness_loss)
        
        prob = cp.Problem(obj, constraints)
        # SCS is great for TV problems
        prob.solve(solver=cp.SCS, max_iters=1500, verbose=False)
        
        if X_dct.value is not None:
            current_img = np.clip(reconstructed_img.value, 0, 255)
            current_psnr = psnr(ground_truth, current_img, data_range=255)
            
            print(f"Trial {i+1:2}: {current_psnr:.2f} dB (Lam: {test_lam:.4f}, TV: {test_tv:.4f})")
            
            if current_psnr > config["BEST_PSNR_FOUND"]:
                print(f"   >>> NEW BEST FOUND! <<<")
                config["BEST_PSNR_FOUND"] = current_psnr
                config["LAMBDA_MIN"] = test_lam * 0.8
                config["LAMBDA_MAX"] = test_lam * 1.2
                config["TV_WEIGHT"] = test_tv # Update TV preference
                best_img = current_img
                improved = True
            
            if best_img is None:
                best_img = current_img

    if improved:
        save_settings(config)
    
    return best_img

# --- 4. EXECUTION ---
if __name__ == "__main__":
    config = load_settings()
    test_size = config["TEST_SIZE"]
    ground_truth = generate_macro_test_image(test_size)
    low_res = ground_truth[::config["SCALE_FACTOR"], ::config["SCALE_FACTOR"]]
    
    reconstructed = enlarge_with_tv_cs(low_res, ground_truth, config)
    
    if reconstructed is not None and config["VISUALIZE"]:
        standard = cv2.resize(low_res, (test_size, test_size), interpolation=cv2.INTER_LINEAR)
        plt.figure(figsize=(15, 6))
        plt.subplot(131); plt.imshow(ground_truth, cmap='viridis'); plt.title("Target")
        plt.subplot(132); plt.imshow(standard, cmap='viridis'); plt.title("Standard")
        plt.subplot(133); plt.imshow(reconstructed, cmap='viridis'); plt.title("TV-CS (Smoother)")
        plt.show()