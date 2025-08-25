import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)

import os
import math
import random
import numpy as np
import pywt
import matplotlib.pyplot as plt
from skimage import io, img_as_float, color
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity as ssim
from skimage.restoration import denoise_tv_chambolle
from concurrent.futures import ThreadPoolExecutor




# ---------------- CONFIGURATION BLOCK ---------------- #
CONFIG = {
    # Paths
    "image_path": "4MP_FLWR GRY.png",
    "out_dir": "dataset",

    # Dataset creation
    "sample_fraction": 0.3,
    "grayscale": True,
    "seed": 42,  # random seed for reproducibility

    # Auto-tune / reconstruction
    "initial_guess": {
        'algo': 'tv_admm',
        'wavelet': 'db2',
        'level': 5,
        'n_iter': 100,
        'lam': 0.05,
        'step': 1.2,
        'cont_decay': 0.95,
        'cont_every': 10,
    },

    # Successive halving parameters
    "budget_schedule": (30, 80, 160),  # maps to n_iter for trials
    "pool_size": 2,
    "topk_ratio": 0.3,
    "anneal": True,
    "T0": 0.02,

    # Display
    "show_images": True,

    # Parallelization
    "max_threads": 8,  # limit ThreadPoolExecutor if desired
}
# ---------------- END CONFIGURATION ---------------- #

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

    # Save previews exactly as generated
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

# ---------------- Wavelet Utilities ---------------- #

def wavelet_soft_threshold(img, wavelet, level, lam):
    coeffs = pywt.wavedec2(img, wavelet, level=level, mode='periodization')
    arr, slices = pywt.coeffs_to_array(coeffs)
    arr = np.sign(arr) * np.maximum(np.abs(arr) - lam, 0.0)
    coeffs = pywt.array_to_coeffs(arr, slices, output_format='wavedec2')
    return pywt.waverec2(coeffs, wavelet, mode='periodization')

def wavelet_weighted_soft_threshold(img, wavelet, level, lam_vec):
    coeffs = pywt.wavedec2(img, wavelet, level=level, mode='periodization')
    arr, slices = pywt.coeffs_to_array(coeffs)
    arr = np.sign(arr) * np.maximum(np.abs(arr) - lam_vec, 0.0)
    coeffs = pywt.array_to_coeffs(arr, slices, output_format='wavedec2')
    return pywt.waverec2(coeffs, wavelet, mode='periodization')

# ---------------- Reconstruction Algorithms ---------------- #

def sparse_reconstruct_ista(measurements, mask, wavelet='db1', level=2, n_iter=80, lam=0.05,
                            step=1.0, tol=1e-6, cont_decay=0.95, cont_every=10):
    """
    ISTA with continuation (lambda decay) and early stopping.
    Data term: 0.5 || M ⊙ (x - y) ||^2
    """
    x = measurements.copy()
    m = mask.astype(float)
    lam_curr = lam
    prev = x
    for it in range(1, n_iter + 1):
        grad = m * (x - measurements)
        x = x - step * grad
        x = wavelet_soft_threshold(x, wavelet, level, lam_curr * step)
        x = np.clip(x, 0, 1)

        # continuation
        if it % cont_every == 0:
            lam_curr *= cont_decay

        # early stopping
        rel = np.linalg.norm(x - prev) / (np.linalg.norm(prev) + 1e-12)
        prev = x
        if rel < tol:
            break
    return x

def sparse_reconstruct_fista(measurements, mask, wavelet='db1', level=2, n_iter=80, lam=0.05,
                             step=1.0, tol=1e-6, cont_decay=0.95, cont_every=10):
    """
    FISTA (accelerated ISTA) with continuation & early stopping.
    """
    m = mask.astype(float)
    x = measurements.copy()
    y = x.copy()
    t = 1.0
    lam_curr = lam
    prev = x
    for it in range(1, n_iter + 1):
        grad = m * (y - measurements)
        x_new = wavelet_soft_threshold(y - step * grad, wavelet, level, lam_curr * step)
        x_new = np.clip(x_new, 0, 1)
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        y = x_new + ((t - 1) / t_new) * (x_new - x)
        x, t = x_new, t_new

        if it % cont_every == 0:
            lam_curr *= cont_decay

        rel = np.linalg.norm(x - prev) / (np.linalg.norm(prev) + 1e-12)
        prev = x
        if rel < tol:
            break
    return x

def sparse_reconstruct_reweighted_l1(measurements, mask, wavelet='db1', level=2,
                                     n_outer=3, n_iter=40, lam=0.05, step=1.0, eps=1e-3):
    """
    Reweighted-ℓ1: outer loops update weights, inner ISTA with weighted shrinkage.
    """
    x = measurements.copy()
    m = mask.astype(float)
    w = None
    for _ in range(n_outer):
        # weights from current coefficients
        coeffs = pywt.wavedec2(x, wavelet, level=level, mode='periodization')
        arr, slices = pywt.coeffs_to_array(coeffs)
        if w is None:
            w = np.ones_like(arr)
        else:
            w = 1.0 / (np.abs(arr) + eps)

        z = x.copy()
        for _ in range(n_iter):
            z = z - m * (z - measurements)  # step=1 in data term
            coeffs = pywt.wavedec2(z, wavelet, level=level, mode='periodization')
            a, s = pywt.coeffs_to_array(coeffs)
            a = np.sign(a) * np.maximum(np.abs(a) - lam * w, 0.0)
            coeffs = pywt.array_to_coeffs(a, s, output_format='wavedec2')
            z = pywt.waverec2(coeffs, wavelet, mode='periodization')
            z = np.clip(z, 0, 1)
        x = z
    return x

def tv_admm_inpaint(measurements, mask, lam_tv=0.1, rho=0.5, n_iter=100):
    """
    PnP-ADMM with TV prox (Chambolle). Minimizes:
      0.5||M ⊙ (x - y)||^2 + lam_tv * TV(x)
    """
    x = measurements.copy()
    v = x.copy()
    u = np.zeros_like(x)
    M = mask.astype(float)

    for _ in range(n_iter):
        # x-update: (M + rho I) x = M*y + rho (v - u)
        rhs = M * measurements + rho * (v - u)
        denom = M + rho
        x = rhs / (denom + 1e-8)

        # v-update: TV prox
        v = denoise_tv_chambolle(x + u, weight=lam_tv / max(rho, 1e-8))

        # dual update
        u = u + x - v

    return np.clip(x, 0, 1)

# ---------------- Channel-wise Reconstruction ---------------- #

def reconstruct_single(channel, algo, params):
    mask = (channel > 0).astype(float)
    if algo == 'fista':
        return sparse_reconstruct_fista(channel, mask,
                                        wavelet=params.get('wavelet', 'db1'),
                                        level=params.get('level', 2),
                                        n_iter=params.get('n_iter', 80),
                                        lam=params.get('lam', 0.05),
                                        step=params.get('step', 1.0),
                                        tol=params.get('tol', 1e-6),
                                        cont_decay=params.get('cont_decay', 0.95),
                                        cont_every=params.get('cont_every', 10))
    elif algo == 'reweighted':
        return sparse_reconstruct_reweighted_l1(channel, mask,
                                                wavelet=params.get('wavelet', 'db1'),
                                                level=params.get('level', 2),
                                                n_outer=params.get('n_outer', 3),
                                                n_iter=params.get('n_iter', 40),
                                                lam=params.get('lam', 0.05),
                                                step=params.get('step', 1.0))
    elif algo == 'tv_admm':
        return tv_admm_inpaint(channel, mask,
                               lam_tv=params.get('lam_tv', 0.1),
                               rho=params.get('rho', 0.5),
                               n_iter=params.get('n_iter', 100))
    else:  # 'ista'
        return sparse_reconstruct_ista(channel, mask,
                                       wavelet=params.get('wavelet', 'db1'),
                                       level=params.get('level', 2),
                                       n_iter=params.get('n_iter', 80),
                                       lam=params.get('lam', 0.05),
                                       step=params.get('step', 1.0),
                                       tol=params.get('tol', 1e-6),
                                       cont_decay=params.get('cont_decay', 0.95),
                                       cont_every=params.get('cont_every', 10))

def reconstruct_channels(masked, algo='fista', **params):
    if masked.ndim == 3 and masked.shape[2] == 3:
        def worker(i):
            return reconstruct_single(masked[:, :, i], algo, params)
        with ThreadPoolExecutor() as executor:
            channels = list(executor.map(worker, range(3)))
        return np.stack(channels, axis=2)
    else:
        if masked.ndim == 3 and masked.shape[2] == 1:
            masked = masked[..., 0]
        return reconstruct_single(masked, algo, params)

# ---------------- Main Processing Function ---------------- #

def process_image(original_path="dataset/original.npy", masked_path="dataset/sampled.npy",
                  algo='fista', show=False, **params):
    original, masked = load_and_match_shapes(original_path, masked_path)
    reconstructed_img = reconstruct_channels(masked, algo=algo, **params)

    # metrics
    mse = mean_squared_error(original, reconstructed_img)
    psnr = peak_signal_noise_ratio(original, reconstructed_img, data_range=1.0)
    ssim_val = ssim(original, reconstructed_img, data_range=1.0)

    if show:
        plt.figure(figsize=(6,6))
        plt.imshow(reconstructed_img, cmap='gray' if reconstructed_img.ndim==2 else None)
        plt.title(f"Reconstructed ({algo})")
        plt.axis('off')
        plt.show()
    
    return reconstructed_img, mse, psnr, ssim_val

# ---------------- Tuning: scoring & proposal ---------------- #

def score_objective(mse, psnr, ssim_val, alpha=0.7, beta=0.2, gamma=0.1):
    """
    Lower is better. Combine MSE (lower), PSNR/SSIM (higher).
    """
    return alpha * mse - beta * (psnr / 100.0) - gamma * ssim_val

def sample_params(seed=None):
    """
    Randomly sample an algorithm + params. If a seed dict is given, jitter it.
    """
    if seed is not None and random.random() < 0.7:
        # Jitter
        algo = seed['algo']
        p = dict(seed)
        p.pop('algo', None)
        if algo in ('ista', 'fista', 'reweighted'):
            p['wavelet'] = random.choice(['db1','db2','sym2','coif1'])
            p['level'] = max(1, p.get('level', 2) + random.randint(-1, 1))
            p['n_iter'] = max(20, p.get('n_iter', 80) + random.randint(-30, 30))
            p['lam'] = max(1e-4, p.get('lam', 0.05) * 10**random.uniform(-0.2, 0.2))
            p['step'] = max(0.2, min(2.0, p.get('step', 1.0) * 10**random.uniform(-0.2, 0.2)))
            if algo in ('ista','fista'):
                p['cont_decay'] = min(0.99, max(0.85, p.get('cont_decay', 0.95) + random.uniform(-0.03, 0.03)))
                p['cont_every'] = max(5, p.get('cont_every', 10) + random.randint(-2, 2))
            if algo == 'reweighted':
                p['n_outer'] = max(2, p.get('n_outer', 3) + random.randint(-1, 1))
        else:  # tv_admm
            p['lam_tv'] = max(1e-3, p.get('lam_tv', 0.1) * 10**random.uniform(-0.5, 0.5))
            p['rho'] = max(0.1, min(2.0, p.get('rho', 0.5) * 10**random.uniform(-0.3, 0.3)))
            p['n_iter'] = max(20, p.get('n_iter', 100) + random.randint(-30, 30))
        return {'algo': algo, **p}

    # Fresh sample
    if random.random() < 0.5:
        algo = random.choice(['fista','ista','reweighted'])
        params = {
            'wavelet': random.choice(['db1','db2','sym2','coif1']),
            'level': random.randint(1, 4),
            'n_iter': random.randint(40, 150),
            'lam': 10**random.uniform(-3, -1),   # log-uniform
            'step': random.uniform(0.5, 1.8),
        }
        if algo in ('ista','fista'):
            params['cont_decay'] = random.uniform(0.88, 0.98)
            params['cont_every'] = random.randint(6, 12)
        else:  # reweighted
            params['n_outer'] = random.randint(2, 4)
        return {'algo': algo, **params}
    else:
        algo = 'tv_admm'
        params = {
            'lam_tv': 10**random.uniform(-2.3, -0.2),
            'rho': random.uniform(0.2, 1.5),
            'n_iter': random.randint(40, 150),
        }
        return {'algo': algo, **params}

# ---------------- Successive Halving + Simulated Annealing ---------------- #

# ---------------- Successive Halving + Simulated Annealing (modified) ---------------- #

def successive_halving(image_paths, budget_schedule=(20, 60, 140), pool_size=24, topk_ratio=0.25,
                      anneal=True, T0=0.02, initial_params=None):
    """
    image_paths: (original_path, masked_path)
    initial_params: dict of starting parameters (used exactly as first trial)
    Returns: (best_img, best_mse, best_psnr, best_ssim, best_params)
    """
    original_path, masked_path = image_paths
    survivors = None
    best_tuple = (None, float('inf'), float('-inf'), float('-inf'), None)  # img, mse, psnr, ssim, params

    for b in budget_schedule:
        if survivors is None:
            # FIRST ROUND: include initial_params exactly as-is
            params_list = []
            if initial_params is not None:
                p = dict(initial_params)
                params_list.append(p)
            while len(params_list) < pool_size:
                params_list.append(sample_params())
        else:
            # Keep survivors and refill with jittered samples
            params_list = survivors[:]
            while len(params_list) < pool_size:
                seed = random.choice(survivors)
                params_list.append(sample_params(seed))

        results = []
        for idx, p in enumerate(params_list, 1):
            # set budget → n_iter for the chosen algo
            p = dict(p)
            p['n_iter'] = b

            try:
                img, mse, psnr, ssim_val = process_image(original_path, masked_path, **p, show=False)
                sc = score_objective(mse, psnr, ssim_val)
                results.append((sc, mse, psnr, ssim_val, p, img))
                print(f"[Budget {b}] Trial {idx}/{len(params_list)} | "
                      f"Algo={p['algo']} | MSE={mse:.5f} PSNR={psnr:.2f} SSIM={ssim_val:.4f} | Params={p}")
                # Track global best
                if mse < best_tuple[1]:
                    best_tuple = (img, mse, psnr, ssim_val, p)
            except Exception as e:
                print(f"[Budget {b}] Trial {idx} failed: {e}")

        # Rank by composite score (lower is better)
        results.sort(key=lambda t: t[0])
        k = max(1, int(len(results) * topk_ratio))
        survivors = [results[i][4] for i in range(k)]  # keep top params

        # Simulated annealing step around the best of this round (optional)
        if anneal and survivors:
            T = T0
            curr = survivors[0]
            curr_img, curr_mse, curr_psnr, curr_ssim = process_image(original_path, masked_path, **curr, show=False)
            for _ in range(6):  # a few anneal moves
                cand = sample_params(curr)
                cand['n_iter'] = b  # keep same budget
                try:
                    img, mse, psnr_v, ssim_v = process_image(original_path, masked_path, **cand, show=False)
                    if mse < curr_mse or np.random.rand() < math.exp((curr_mse - mse) / max(T, 1e-12)):
                        curr, curr_img, curr_mse, curr_psnr, curr_ssim = cand, img, mse, psnr_v, ssim_v
                        if mse < best_tuple[1]:
                            best_tuple = (img, mse, psnr_v, ssim_v, cand)
                except Exception:
                    pass
                T *= 0.9
            survivors[0] = curr  # replace with annealed best

    return best_tuple  # (img, mse, psnr, ssim, params)

# ---------------- High-level Auto-Tune ---------------- #

def stochastic_auto_tune(image_path, out_dir="dataset", sample_fraction=0.4, grayscale=True,
                         initial_params=None, n_trials=50):
    # Always remake dataset
    make_dataset(image_path, out_dir=out_dir, sample_fraction=sample_fraction, grayscale=grayscale)
    original_path = os.path.join(out_dir, "original.npy")
    masked_path = os.path.join(out_dir, "sampled.npy")

    # If user gives initial params, seed the pool with them
    seed_pool = []
    if initial_params is not None:
        p = dict(initial_params)
        algo = p.pop('algo', 'fista')
        seed_pool.append({'algo': algo, **p})

    # Run Successive Halving (+ anneal) across algorithms/params
    best_img, best_mse, best_psnr, best_ssim, best_params = successive_halving(
        (original_path, masked_path),
        pool_size=CONFIG["pool_size"],
        budget_schedule=CONFIG["budget_schedule"],
        topk_ratio=CONFIG["topk_ratio"],
        anneal=CONFIG["anneal"],
        T0=CONFIG["T0"]
    )

    if best_img is not None:
        print("\n=== Best Parameters Found ===")

    # Make a full template with all possible keys
        full_template = CONFIG["initial_guess"].copy()
    # Merge algorithm-specific defaults for other algos
        algo_defaults = {
            'ista': {'wavelet':'db1','level':2,'n_iter':100,'lam':0.05,'step':1.0,'cont_decay':0.95,'cont_every':10},
            'fista': {'wavelet':'db1','level':2,'n_iter':100,'lam':0.05,'step':1.0,'cont_decay':0.95,'cont_every':10},
            'reweighted': {'wavelet':'db1','level':2,'n_outer':3,'n_iter':40,'lam':0.05,'step':1.0},
            'tv_admm': {'lam_tv':0.1,'rho':0.5,'n_iter':100}
        }
    # Update template with algo-specific defaults
        full_template.update(algo_defaults.get(best_params['algo'], {}))
    # Overwrite with best_params
        full_template.update(best_params)

        print(full_template)
        print(f"Best MSE = {best_mse:.6f}")
        print(f"Best PSNR = {best_psnr:.2f} dB")
        print(f"Best SSIM = {best_ssim:.4f}")

        # Load original and sampled (as generated by dataset)
        original_img_path = os.path.join(out_dir, "original.png")
        sampled_img_path = os.path.join(out_dir, "sampled.png")
        original_disp = io.imread(original_img_path)
        sampled_disp = io.imread(sampled_img_path)
        best_disp = np.clip(best_img, 0, 1)

        # Show original, sampled, and reconstructed images side by side
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(original_disp, cmap='gray' if original_disp.ndim==2 else None)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(sampled_disp, cmap='gray' if sampled_disp.ndim==2 else None)
        axes[1].set_title("Sampled")
        axes[1].axis("off")

        axes[2].imshow(best_disp, cmap='gray' if best_disp.ndim==2 else None)
        axes[2].set_title(f"Best Reconstructed ({best_params['algo']})")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()
    else:
        print("No successful reconstructions.")

# ---------------- Run Example ---------------- #

if __name__ == "__main__":
    # Example: you can include an initial guess (algo optional)
    initial_guess = {
        'algo': 'tv_admm',
        'wavelet': 'db2',
        'level': 5,
        'n_iter': 100,
        'lam': 0.05,
        'step': 1.2,
        'cont_decay': 0.95,
        'cont_every': 10,
    }

    stochastic_auto_tune(
        image_path=CONFIG["image_path"],
        out_dir=CONFIG["out_dir"],
        sample_fraction=CONFIG["sample_fraction"],
        grayscale=CONFIG["grayscale"],
        initial_params=CONFIG["initial_guess"],
        n_trials=1  # kept for interface compatibility (not used by halving)
    )
