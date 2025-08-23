import sys
import os
import numpy as np
import pywt
import matplotlib.pyplot as plt
from skimage import io, img_as_float, color
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio

print("Python executable:", sys.executable)
print("Python version:", sys.version)

# ---------------- Dataset Creation ---------------- #

def make_dataset(image_path, out_dir="dataset", sample_fraction=0.4, seed=42, grayscale=True):
    """
    Create a compressed sensing dataset from an input image.
    """
    if seed is not None:
        np.random.seed(seed)

    # Load image and convert to float [0,1]
    img = img_as_float(io.imread(image_path))

    # Convert to grayscale if requested
    if grayscale and img.ndim == 3:
        # If image has alpha channel, drop it
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img = color.rgb2gray(img)
        img = img[..., np.newaxis]  # keep 3D shape

    h, w = img.shape[:2]
    total_pixels = h * w

    # Random mask
    mask = np.zeros((h, w), dtype=bool)
    num_samples = int(total_pixels * sample_fraction)
    coords = np.random.choice(total_pixels, num_samples, replace=False)
    mask.flat[coords] = True

    # Create sampled image
    sampled_img = np.zeros_like(img)
    sampled_img[mask] = img[mask]

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Save numpy arrays
    np.save(os.path.join(out_dir, "original.npy"), img)
    np.save(os.path.join(out_dir, "sampled.npy"), sampled_img)

    # Save preview PNGs
    if grayscale:
        io.imsave(os.path.join(out_dir, "original.png"), (img.squeeze() * 255).astype(np.uint8))
        io.imsave(os.path.join(out_dir, "sampled.png"), (sampled_img.squeeze() * 255).astype(np.uint8))
    else:
        io.imsave(os.path.join(out_dir, "original.png"), (img * 255).astype(np.uint8))
        io.imsave(os.path.join(out_dir, "sampled.png"), (sampled_img * 255).astype(np.uint8))

    print(f"Dataset created in '{out_dir}/'")

# ---------------- Wavelet Utilities ---------------- #

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
    for i in range(n_iter):
        # Enforce mask
        x = x + mask * (measurements - x)

        # Wavelet transform
        coeffs = wavelet_decompose(x, wavelet, level)
        arr, slices = coeffs_to_array(coeffs)

        # Soft-threshold
        arr = np.sign(arr) * np.maximum(np.abs(arr) - lam, 0)

        # Inverse wavelet
        coeffs = array_to_coeffs(arr, slices)
        x = wavelet_reconstruct(coeffs, wavelet)

        # Clip
        x = np.clip(x, 0, 1)

    return x

def interpolate_channel(channel, mask, wavelet='db1', level=2, n_iter=50, lam=0.1, step=1.0):
    return sparse_reconstruct_ista(channel, mask, wavelet, level, n_iter, lam, step)

# ---------------- Main Processing Function ---------------- #

def process_image(original_path, masked_path,
                  wavelet='db1', level=2, n_iter=50, lam=0.1, step=1.0):
    # Load or create dataset if missing
    if not os.path.exists(original_path) or not os.path.exists(masked_path):
        print("Dataset not found. Creating dataset...")
        make_dataset(image_path="1MP GRY.png", out_dir="dataset",
                     sample_fraction=0.4, seed=42, grayscale=True)
        original_path = "dataset/original.npy"
        masked_path = "dataset/sampled.npy"

    original = np.load(original_path)
    masked = np.load(masked_path)

    # Handle RGB or grayscale
    if original.ndim == 3 and original.shape[2] == 3:
        print("Processing RGB image...")
        channels = []
        for i, color_name in enumerate(['Red','Green','Blue']):
            print(f"Reconstructing {color_name} channel...")
            channel = masked[:, :, i]
            mask = (channel > 0).astype(float)
            reconstructed = interpolate_channel(channel, mask, wavelet, level, n_iter, lam, step)
            channels.append(reconstructed)
        reconstructed_img = np.stack(channels, axis=2)
    else:
        print("Processing Grayscale image...")
        if masked.ndim == 3 and masked.shape[2] == 1:
            masked = masked[..., 0]
            original = original[..., 0]
        mask = (masked > 0).astype(float)
        reconstructed_img = interpolate_channel(masked, mask, wavelet, level, n_iter, lam, step)

    # Display
    fig, axs = plt.subplots(1,2, figsize=(10,5))
    axs[0].imshow(original, cmap='gray' if original.ndim==2 else None)
    axs[0].set_title("Original")
    axs[0].axis('off')

    axs[1].imshow(reconstructed_img, cmap='gray' if original.ndim==2 else None)
    axs[1].set_title("Reconstructed")
    axs[1].axis('off')
    plt.show()

    # Accuracy
    mse = mean_squared_error(original, reconstructed_img)
    psnr = peak_signal_noise_ratio(original, reconstructed_img, data_range=1.0)
    print(f"Reconstruction Accuracy:\n  MSE = {mse:.6f}\n  PSNR = {psnr:.2f} dB")

    return reconstructed_img

# ---------------- Adjustable Parameters ---------------- #
params = {
    "wavelet": 'db1',
    "level": 2,
    "n_iter": 30,
    "lam": 0.05,
    "step": 1.0,
    "sample_fraction": 0.4
}

# ---------------- Run ---------------- #
if __name__ == "__main__":
    reconstructed = process_image(
        original_path="dataset/original.npy",
        masked_path="dataset/sampled.npy",
        wavelet=params["wavelet"],
        level=params["level"],
        n_iter=params["n_iter"],
        lam=params["lam"],
        step=params["step"]
    )
