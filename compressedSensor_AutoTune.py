import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)

import os
import random
import numpy as np
import pywt
import matplotlib.pyplot as plt
from skimage import io, img_as_float, color
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
from concurrent.futures import ThreadPoolExecutor

# ---------------- Dataset Creation ---------------- #

def make_dataset(image_path, out_dir="dataset", sample_fraction=0.4, seed=42, grayscale=True):
    np.random.seed(seed)
    img = img_as_float(io.imread(image_path))
    
    # Convert to grayscale if requested
    if grayscale and img.ndim == 3:
        if img.shape[2] == 4:  # RGBA
            img = img[..., :3]
        img = color.rgb2gray(img)[..., np.newaxis]

    h, w = img.shape[:2]
    total_pixels = h * w
    mask = np.zeros((h, w), dtype=bool)
    num_samples = int(total_pixels * sample_fraction)
    coords = np.random.choice(total_pixels, num_samples, replace=False)
    mask.flat[coords] = True
    sampled_img = np.zeros_like(img)
    sampled_img[mask] = img[mask]

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "original.npy"), img)
    np.save(os.path.join(out_dir, "sampled.npy"), sampled_img)

    # Save previews
    if grayscale:
        io.imsave(os.path.join(out_dir, "original.png"), (img.squeeze() * 255).astype(np.uint8))
        io.imsave(os.path.join(out_dir, "sampled.png"), (sampled_img.squeeze() * 255).astype(np.uint8))
    else:
        io.imsave(os.path.join(out_dir, "original.png"), (img * 255).astype(np.uint8))
        io.imsave(os.path.join(out_dir, "sampled.png"), (sampled_img * 255).astype(np.uint8))

    print(f"Dataset created in '{out_dir}'.")

# ---------------- Load and Match Shapes ---------------- #

def load_and_match_shapes(original_path, masked_path):
    original = np.load(original_path)
    masked = np.load(masked_path)

    # Remove singleton channel for grayscale
    if original.ndim == 3 and original.shape[2] == 1:
        original = original[..., 0]
    if masked.ndim == 3 and masked.shape[2] == 1:
        masked = masked[..., 0]

    # Crop to minimum matching dimensions
    h = min(original.shape[0], masked.shape[0])
    w = min(original.shape[1], masked.shape[1])
    original = original[:h, :w]
    masked = masked[:h, :w]

    return original, masked

# ---------------- Wavelet Functions ---------------- #

def wavelet_decompose(image, wavelet='db1', level=2):
    return pywt.wavedec2(image, wavelet, level=level, mode='periodization')

def wavelet_reconstruct(coeffs, wavelet='db1'):
    return pywt.waverec2(coeffs, wavelet, mode='periodization')

def coeffs_to_array(coeffs):
    arr, slices = pywt.coeffs_to_array(coeffs)
    return arr, slices

def array_to_coeffs(arr, slices):
    return pywt.array_to_coeffs(arr, slices, output_format='wavedec2')

# ---------------- Sparse Reconstruction (ISTA) ---------------- #

def sparse_reconstruct_ista(measurements, mask, wavelet='db1', level=2, n_iter=50, lam=0.1, step=1.0):
    x = np.copy(measurements)
    mask = mask.astype(float)
    for _ in range(n_iter):
        x = x + mask * (measurements - x)
        coeffs = wavelet_decompose(x, wavelet, level)
        arr, slices = coeffs_to_array(coeffs)
        arr = np.sign(arr) * np.maximum(np.abs(arr) - lam, 0)
        coeffs = array_to_coeffs(arr, slices)
        x = wavelet_reconstruct(coeffs, wavelet)
        x = np.clip(x, 0, 1)
    return x

def interpolate_channel(channel, mask, wavelet='db1', level=2, n_iter=50, lam=0.1, step=1.0):
    return sparse_reconstruct_ista(channel, mask, wavelet, level, n_iter, lam, step)

# ---------------- Multithreaded Reconstruction ---------------- #

def reconstruct_channels(masked, wavelet='db1', level=2, n_iter=50, lam=0.1, step=1.0):
    if masked.ndim == 3 and masked.shape[2] == 3:
        def worker(i):
            channel = masked[:, :, i]
            mask = (channel > 0).astype(float)
            return interpolate_channel(channel, mask, wavelet, level, n_iter, lam, step)
        with ThreadPoolExecutor() as executor:
            channels = list(executor.map(worker, range(3)))
        return np.stack(channels, axis=2)
    else:
        if masked.ndim == 3 and masked.shape[2] == 1:
            masked = masked[..., 0]
        mask = (masked > 0).astype(float)
        return interpolate_channel(masked, mask, wavelet, level, n_iter, lam, step)

# ---------------- Main Processing Function ---------------- #

def process_image(original_path="dataset/original.npy", masked_path="dataset/sampled.npy",
                  wavelet='db1', level=2, n_iter=50, lam=0.1, step=1.0, show=False):
    original, masked = load_and_match_shapes(original_path, masked_path)
    reconstructed_img = reconstruct_channels(masked, wavelet, level, n_iter, lam, step)
    
    mse = mean_squared_error(original, reconstructed_img)
    psnr = peak_signal_noise_ratio(original, reconstructed_img, data_range=1.0)
    
    if show:
        plt.figure(figsize=(6,6))
        plt.imshow(reconstructed_img, cmap='gray' if reconstructed_img.ndim==2 else None)
        plt.title("Reconstructed Image")
        plt.axis('off')
        plt.show()
    
    return reconstructed_img, mse, psnr

# ---------------- Stochastic Auto-Tuning ---------------- #

def stochastic_auto_tune(image_path, out_dir="dataset", sample_fraction=0.4, grayscale=True,
                         initial_params=None, n_trials=50):
    # Always remake dataset
    make_dataset(image_path, out_dir=out_dir, sample_fraction=sample_fraction, grayscale=grayscale)
    original_path = os.path.join(out_dir, "original.npy")
    masked_path = os.path.join(out_dir, "sampled.npy")

    best_mse = float('inf')
    best_psnr = float('-inf')  # Track PSNR corresponding to best MSE
    best_result = None
    best_params = None

    for i in range(n_trials):
        if initial_params:
            candidate_params = {
                "wavelet": random.choice(["db1", "db2", "haar", "sym2"]),
                "level": max(1, initial_params["level"] + random.randint(-1, 1)),
                "n_iter": max(10, initial_params["n_iter"] + random.randint(-20, 20)),
                "lam": max(1e-4, initial_params["lam"] + random.uniform(-0.01, 0.01)),
                "step": max(0.05, initial_params["step"] + random.uniform(-0.2, 0.2)),
            }
        else:
            candidate_params = {
                "wavelet": random.choice(["db1", "db2", "haar", "sym2"]),
                "level": random.randint(1, 4),
                "n_iter": random.randint(50, 200),
                "lam": random.uniform(0.001, 0.1),
                "step": random.uniform(0.1, 2.0),
            }

        try:
            img, mse, psnr = process_image(original_path, masked_path, **candidate_params, show=False)

            print(f"Trial {i+1}/{n_trials}: MSE={mse:.4f}, PSNR={psnr:.2f} dB, params={candidate_params}")

            if mse < best_mse:
                best_mse = mse
                best_psnr = psnr
                best_result = img
                best_params = candidate_params

        except Exception as e:
            print(f"Trial {i+1}/{n_trials} failed with error: {e}")

    if best_result is not None:
        print("\nBest Parameters Found:")
        print(best_params)
        print(f"Best MSE={best_mse:.4f}")
        print(f"PSNR at Best MSE={best_psnr:.2f} dB")  # <-- Added PSNR reporting

        # Load original for display
        original = np.load(original_path)

        # Ensure values are in [0,1]
        original_disp = np.clip(original, 0, 1)
        best_disp = np.clip(best_result, 0, 1)

        # Show the final result side by side
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(original_disp, cmap='gray' if original_disp.ndim==2 else None)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(best_disp, cmap='gray' if best_disp.ndim==2 else None)
        axes[1].set_title("Best Reconstructed")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()
    else:
        print("No successful reconstructions.")

# ---------------- Run Example ---------------- #

if __name__ == "__main__":
    stochastic_auto_tune(
        image_path="1MP GRY.png",
        out_dir="dataset",
        sample_fraction=0.4,
        grayscale=True,
        initial_params={'wavelet':'sym2','level':7,'n_iter':67,'lam':0.06516217782694049,'step':1.2936710717113946},
        n_trials=50
    )
