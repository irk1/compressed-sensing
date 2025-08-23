import numpy as np
from skimage import io, img_as_float, color
import os

def make_dataset(image_path, out_dir="dataset", sample_fraction=0.4, seed=None, grayscale=True):
    """
    Create a dataset for compressed sensing testing.

    Parameters
    ----------
    image_path : str
        Path to input image file.
    out_dir : str
        Directory where outputs will be saved.
    sample_fraction : float
        Fraction of pixels to keep (0 < sample_fraction < 1).
    seed : int, optional
        Random seed for reproducibility.
    grayscale : bool
        If True, converts image to grayscale before processing.
    """
    if seed is not None:
        np.random.seed(seed)

    # Load image and convert to float [0,1]
    img = img_as_float(io.imread(image_path))

    # Convert RGBA to RGB if needed
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    # Convert to grayscale if requested
    if grayscale:
        if img.ndim == 3:
            img = color.rgb2gray(img)
        # expand dims to keep shape consistent
        img = img[..., np.newaxis]

    # Get shape info
    h, w = img.shape[:2]
    total_pixels = h * w

    # Random mask
    mask = np.zeros((h, w), dtype=bool)
    num_samples = int(total_pixels * sample_fraction)
    coords = np.random.choice(total_pixels, num_samples, replace=False)
    mask.flat[coords] = True

    # Create sampled version
    sampled_img = np.zeros_like(img)
    sampled_img[mask] = img[mask]

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Save numpy arrays (overwrite if exist)
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
    print(f"  - original.npy (ground truth)")
    print(f"  - sampled.npy (masked image)")
    print(f"  - original.png (preview)")
    print(f"  - sampled.png (preview)")

# ---------------- Example Usage ---------------- #
if __name__ == "__main__":
    make_dataset("256x256 GRY.png", sample_fraction=0.6, seed=42, grayscale=True)
