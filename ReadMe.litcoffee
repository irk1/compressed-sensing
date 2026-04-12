
### CSNEW-Image_Gen: High-Precision Compressed Sensing Upscaler (v5.1)

CSNEW-Image_Gen is a professional-grade image reconstruction engine designed for high-bit-depth digital negatives. Unlike standard AI upscalers that generate pixels via neural networks, this tool uses Compressed Sensing and Basis Pursuit Denoising to reconstruct signals using mathematical optimization.

The system is optimized for Dual-Xeon Workstations, leveraging high core counts, shared memory blocks, and massive RAM pools to solve complex convex optimization problems at 16-bit precision.

---

### Basis of Function

The core of the program is based on the premise that natural images are sparse in a transform domain, specifically the Discrete Cosine Transform (DCT).

#### 1. Mathematical Objective
The solver reconstructs an image $x$ by minimizing a multi-objective cost function:

$$\min_{x} \lambda \| \Psi x \|_1 + \gamma \text{TV}(x)$$

Subject to the constraint:
$$\| Ax - y \|_2 \leq \epsilon$$

* $\Psi$: The DCT transform operator.
* $\| \cdot \|_1$: $L_1$ Norm; promotes sparsity, ensuring only the most significant image structures are preserved.
* $\text{TV}(x)$: Total Variation regularization to reduce noise while preserving sharp edges.
* $A$: The downsampling operator representing the relationship between the desired high-res image and the low-res sensor data $y$.

#### 2. The 32-Bit Float Pipeline
To avoid the precision loss inherent in 8-bit or 16-bit integer processing, the engine operates in 32-bit Floating Point space. This allows the ECOS solver to converge rapidly within its 500-iteration limit, preserving the full dynamic range of RAW sensor data.

---

### Architecture & Portability

CSNEW-Image_Gen is designed for high-efficiency workstation deployment.

* **MonolithDNGWriter (Native DNG 1.7.1.0):** Completely removes the reliance on external ExifTool dependencies. The engine natively packs and structures 16-bit binary pixel data directly into compliant DNG containers with injected color matrices and neutral illuminant tags.
* **Shared Memory Orchestration:** Utilizes `multiprocessing.sharedctypes` to map multi-gigabyte 32-bit float image arrays directly into RAM. This prevents the severe memory overhead of passing large numpy arrays between standard Python processes.
* **Hardware Safety Gate:** Real-time monitoring of system RAM (via `psutil`) pauses CPU dispatching if free memory falls below the user-defined threshold in `settings.json`.
* **Live Telemetry:** Writes asynchronous output to `training_results.csv`, logging completion times, parameters, and success states for long-term analytics.

---

### Supported File Formats

The engine utilizes `rawpy` to maintain a high-precision 16-bit signal chain from sensor to final export.

#### Supported Input Formats:
* **Digital Negatives (RAW):** `.dng`, `.nef`, `.cr2`, `.arw`, `.orf`.
* **Standard Raster:** `.tif` / `.tiff` (preferred), `.png`, `.jpg`.

#### Output Formats:
* **TIFF (Default):** 16-bit, Zip-compressed (32946) standard TIFF.
* **DNG (Optional):** Using the `--dng` flag, it exports to a 16-bit Uncompressed Digital Negative, preserving reconstruction data in a format natively compatible with Adobe Camera Raw.

---

### How to Use

The engine is distributed as a command-line utility.

`./CSNEW-Image_Gen [file]`
Process a single image using `settings.json` parameters.

`./CSNEW-Image_Gen --batch [directory]`
Production mode: Processes all images in a directory. Skips already-upscaled files to allow safe resuming of interrupted jobs.

`./CSNEW-Image_Gen --test [file/dir]`
Visual Training Mode: Generates a 9-way parameter sweep grid to manually find the optimal parameters. Outputs are saved as labeled files (e.g., `_L1e-05_T0.002.tif`) for direct evaluation.

`./CSNEW-Image_Gen --tune [directory] [--full]`
Auto-Tuning Engine: Reads a folder of sample images and automatically calculates the mathematical optimum for Lambda and TV weights, writing the winner directly to your `settings.json`. By default, it operates on a fast 512x512 central crop. Append `--full` to bypass the heuristic and tune against the entire image.

`--dng`
Can be appended to any command to output `.dng` files instead of `.tif`.

---

### Configuration
The engine requires a `settings.json` in the execution directory to define hardware and math limits:

```json
{
    "SCALE_FACTOR": 2,
    "TILE_SIZE": 64,
    "OVERLAP": 8,
    "LAMBDA_MIN": 0.0001,
    "TV_WEIGHT": 0.005,
    "MAX_THREADS": 48,
    "MAX_RAM_GB": 30,
    "DENOISE": false
}
```

---

### Auto-Tuning and Scoring Analytics

When running in `--tune` mode, the program utilizes a No-Reference Image Quality metric to quantify reconstruction quality automatically, bypassing the need for subjective human observation.

The engine calculates mathematical fidelity using a Custom Quality Metric ($Q$):

$$Q = \frac{\text{Var}(\nabla^2 I)}{\text{Var}(I - \text{MedianBlur}(I, 3))}$$

* **Numerator (Sharpness):** The variance of the Laplacian ($\nabla^2$). This measures edge crispness and high-frequency botanical details.
* **Denominator (Noise Penalty):** The variance of the difference between the reconstructed image and a median-blurred version of itself. This heavily penalizes the score if the TV weight was too low and left behind grainy ECOS artifacts.

The engine tests a grid of 9 variations, logs the score of each, averages them across your tuning directory, and automatically applies the highest-scoring combination to your settings.

---

### CS_Verify: Reconstruction Fidelity Analytics Suite

The CS_Verify utility is a separate, high-precision diagnostic tool designed to mathematically quantify the success of Compressed Sensing reconstructions against downsampled reference targets. 

### Analytical Metrics

The suite utilizes four primary mathematical benchmarks to evaluate signal integrity and solver convergence.

#### 1. Structural Similarity Index (SSIM)
Unlike simple error measurements, SSIM models the human visual system’s perception of luminance, contrast, and structure.

* Target: $> 0.980$
* Formula:
$$SSIM(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$

#### 2. Peak Signal-to-Noise Ratio (PSNR)
Quantifies the ratio between the maximum possible power of the signal and the power of corrupting noise introduced during the 500-iteration solver process.

* Target: $> 30 \text{ dB}$
* Formula:
$$PSNR = 10 \cdot \log_{10}\left(\frac{MAX_I^2}{MSE}\right)$$

#### 3. Mean Squared Error (MSE)
Represents the cumulative squared error between the reconstructed image and the original reference.

* Target: $\to 0$
* Formula:
$$MSE = \frac{1}{mn} \sum_{i=0}^{m-1} \sum_{j=0}^{n-1} [I(i,j) - K(i,j)]^2$$

#### 4. Error Heatmapping
The program generates a spatial divergence map using the Jet Colormap. This allows for the identification of specific "Tile Boundary" artifacts or areas where the TV Weight may have over-smoothed microscopic textures.

---

### Technical Usage (CS_Verify)

#### Requirements
* Python 3.10+
* Dependencies: `opencv-python`, `skimage`, `matplotlib`, `scipy`

#### Execution Command
`python cs_verify.py <Original_Path> <Reconstructed_Path>`

#### Output Artifacts
1.  **Console Summary:** A quick-look table of MSE, PSNR, Correlation, and SSIM.
2.  **Comparison_Report.pdf:** A professional-grade, multi-page document containing:
    * Side-by-Side Visuals: The reference image vs. the reconstruction.
    * Normalized Difference Map: Highlighting the exact spatial location of residuals.
    * Interpretation Glossary: Definitions for each metric to assist in parameter tuning for the next batch run.