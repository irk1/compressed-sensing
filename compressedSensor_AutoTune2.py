import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from sklearn.linear_model import OrthogonalMatchingPursuit
import time

# ==========================
# Parameters you can change
# ==========================
IMG_SIZE = 64              # size of synthetic test image
SAMPLE_RATIO = 0.3         # % of pixels kept
NUM_TRIALS = 3             # number of trials per hyperparameter step
MAX_ITER_STEPS = [10, 20, 30, 40, 50]  # hyperparameter values to test
SEED = 42

np.random.seed(SEED)

# ==========================
# Dataset creation
# ==========================
# Create synthetic "original" image
original = np.zeros((IMG_SIZE, IMG_SIZE))
# Draw some simple shapes
original[16:48, 24:40] = 1.0  # rectangle
rr, cc = np.ogrid[:IMG_SIZE, :IMG_SIZE]
circle = (rr - 48)**2 + (cc - 16)**2 <= 8**2
original[circle] = 0.5

# Masked (simulated measurements)
mask = np.random.choice([0, 1], size=original.shape, p=[1-SAMPLE_RATIO, SAMPLE_RATIO])
sampled = original * mask

# Save so they're guaranteed to exist
np.save("original.npy", original)
np.save("masked.npy", sampled)

# ==========================
# Compressed sensing solver
# ==========================
def reconstruct(sampled_img, mask, max_iter):
    """
    Very simplified compressed sensing solver using Orthogonal Matching Pursuit
    on a flattened 1D version of the image.
    """
    y = sampled_img.flatten()
    M = np.diag(mask.flatten())

    # Measurement matrix A (here identity masked)
    A = M

    # Fit OMP
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=max_iter)
    omp.fit(A, y)
    x_rec = omp.coef_

    return x_rec.reshape(sampled_img.shape)

# ==========================
# Hyperparameter tuning
# ==========================
best_score = -1
best_params = None
best_rec = None

for max_iter in MAX_ITER_STEPS:
    print(f"\n--- Testing max_iter={max_iter} ---")
    scores = []
    start_time = time.time()
    
    for trial in range(NUM_TRIALS):
        print(f" Trial {trial+1}/{NUM_TRIALS}...")
        rec = reconstruct(sampled, mask, max_iter)
        
        # Compute metrics
        trial_psnr = psnr(original, rec, data_range=1)
        trial_ssim = ssim(original, rec, data_range=1)
        scores.append((trial_psnr, trial_ssim))
    
    avg_psnr = np.mean([s[0] for s in scores])
    avg_ssim = np.mean([s[1] for s in scores])
    elapsed = time.time() - start_time
    
    print(f" Average PSNR={avg_psnr:.4f}, Average SSIM={avg_ssim:.4f}, Time={elapsed:.2f}s")
    
    # Save best
    if avg_psnr > best_score:
        best_score = avg_psnr
        best_params = max_iter
        best_rec = rec

# ==========================
# Final results
# ==========================
print("\n=============================")
print(f"Best max_iter = {best_params}")
print(f"Best PSNR     = {best_score:.4f}")
print("=============================\n")

# ==========================
# Visualization
# ==========================
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(original, cmap="gray")
plt.title("Original")

plt.subplot(1, 3, 2)
plt.imshow(sampled, cmap="gray")
plt.title("Sampled (masked)")

plt.subplot(1, 3, 3)
plt.imshow(best_rec, cmap="gray")
plt.title(f"Reconstructed (iter={best_params})")

plt.tight_layout()
plt.show()
