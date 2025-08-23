import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)

import numpy as np
import pywt
import matplotlib.pyplot as plt
from skimage import io, img_as_float, color
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
import os
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

    # If RGB, crop each channel to the same size
    if original.ndim == 3 and masked.ndim == 3:
        channels = original.shape[2]
        h = min(original.shape[0], masked.shape[0])
        w = min(original.shape[1], masked.shape[1])
        original = original[:h, :w, :channels]
        masked = masked[:h, :w, :channels]

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
    for i in range(n_iter):
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

def process_image(original_path="original.npy", masked_path="sampled.npy",
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

def stochastic_auto_tune(image_path, out_dir="dataset", sample_fraction=0.4,
                         initial_params=None, n_trials=50, noise_scale=None, grayscale=True):
    # Always remake dataset
    make_dataset(image_path, out_dir, sample_fraction, seed=42, grayscale=grayscale)
    original_path = os.path.join(out_dir, "original.npy")
    masked_path = os.path.join(out_dir, "sampled.npy")

    if initial_params is None:
        initial_params = {'wavelet':'db1','level':2,'n_iter':20,'lam':0.1,'step':1.0}
    if noise_scale is None:
        noise_scale = {'level':1,'n_iter':10,'lam':0.05,'step':0.5}

    wavelets = ['db1','db2','sym2','coif1']
    best_params = initial_params.copy()
    best_img, best_mse, best_psnr = process_image(original_path, masked_path, **best_params, show=False)

    for trial in range(n_trials):
        # Random offset from current best
        level = max(1, best_params['level'] + np.random.randint(-noise_scale['level'], noise_scale['level']+1))
        n_iter = max(1, best_params['n_iter'] + np.random.randint(-noise_scale['n_iter'], noise_scale['n_iter']+1))
        lam = max(0.001, best_params['lam'] + np.random.uniform(-noise_scale['lam'], noise_scale['lam']))
        step = max(0.1, best_params['step'] + np.random.uniform(-noise_scale['step'], noise_scale['step']))
        wavelet = np.random.choice(wavelets)

        candidate_params = {'wavelet':wavelet,'level':level,'n_iter':n_iter,'lam':lam,'step':step}
        img, mse, psnr = process_image(original_path, masked_path, **candidate_params, show=False)

        if mse < best_mse:
            best_img = img
            best_mse = mse
            best_psnr = psnr
            best_params = candidate_params.copy()
            print(f"Trial {trial+1}: New best! MSE={best_mse:.6f}, PSNR={best_psnr:.2f} | Params={best_params}")

    print("\n=== Best Parameters Found ===")
    for k,v in best_params.items():
        print(f"{k} = {v}")
    print(f"MSE={best_mse:.6f}, PSNR={best_psnr:.2f}")

    plt.figure(figsize=(6,6))
    plt.imshow(best_img, cmap='gray' if best_img.ndim==2 else None)
    plt.title("Best Stochastic Reconstruction")
    plt.axis('off')
    plt.show()

    return best_img, best_params

# ---------------- Run Example ---------------- #

if __name__ == "__main__":
    # 1. Run stochastic auto-tune on your image
    stochastic_auto_tune(
        image_path="3.3MP_FLWR GRY.png",
        out_dir="dataset",
        sample_fraction=0.4,
        initial_params={'wavelet':'coif1','level':7,'n_iter':58,'lam':0.0681,'step':1.35},
        n_trials=50,
        grayscale=True
    )
