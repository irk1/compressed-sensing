### IRK_CS-Upscale: High-Precision Compressed Sensing Upscaler (v5.8)

IRK_CS-Upscale is a professional-grade image reconstruction engine designed for high-bit-depth digital negatives. Unlike standard AI upscalers that generate pixels via neural networks, this tool uses Compressed Sensing and Basis Pursuit Denoising to reconstruct signals using mathematical optimization.

The system was built and tested on a Dual-Xeon Workstation, leveraging high core counts, shared memory blocks, and massive RAM pools to solve complex convex optimization problems at 16-bit precision; As such this software may not run well on smaller machines pending future optimization and updates.

---

### Basis of Function

The core of the program is based on the premise that natural images are sparse in a transform domain, specifically the Discrete Cosine Transform (DCT).

#### 1. Mathematical Objective

The solver reconstructs an image $x$ by minimizing a multi-objective cost function:

$$\min_{x} \lambda \Vert{} \Psi x \Vert{}_1 + \gamma \text{TV}(x)$$

Subject to the constraint:


$$\Vert{} Ax - y \Vert{}_2 \leq \epsilon$$

* $\Psi$: The DCT transform operator.
* $\Vert{} \cdot \Vert{}_1$: $L_1$ Norm; promotes sparsity, ensuring only the most significant image structures are preserved.
* $\text{TV}(x)$: Total Variation regularization to reduce noise while preserving sharp edges.
* $A$: The downsampling operator representing the relationship between the desired high-res image and the low-res sensor data $y$.

#### 2. The 32-Bit Float Pipeline

To avoid the precision loss inherent in 8-bit or 16-bit integer processing, the engine operates in 32-bit Floating Point space. This allows the ECOS solver to converge rapidly within its 500-iteration limit, preserving the full dynamic range of RAW sensor data.

---

### Architecture & Portability

IRK_CS-Upscale is designed for high-efficiency workstation deployment.

* **MonolithDNGWriter (Native DNG 1.7.1.0):** Completely removes the reliance on external ExifTool dependencies. The engine natively packs and structures 16-bit binary pixel data directly into compliant DNG containers, preserving the full sensor signal chain for RAW/CFA inputs.
* **Tri-Path Hybrid Signal Chain:** Dynamically inspects incoming tensor dimensions and file extensions to route images through the optimal pipeline:
* **Bayer CFA RAWs:** Processed as 4-channel sub-sampled sensor planes (`R, G1, G2, B`).
* **Linear DNGs:** Automatically detected via 3-channel sensor array geometry and routed through the RGB reconstruction pipeline while retaining camera color metadata.
* **Standard Raster Images:** Processed as normalized 32-bit float 3-channel RGB arrays.


* **Native RAW & Color Metadata Synthesis:** For standard `.dng` files, exact rational `ColorMatrix1/2` and `AsShotNeutral` tags are parsed directly from the TIFF header. For proprietary camera RAW containers (such as Olympus `.orf`, `.cr2`, `.nef`, `.arw`), TIFF header parsing is bypassed in favor of `rawpy`/LibRaw metadata extraction—computing green-referenced `AsShotNeutral` white balance ratios, `XYZ -> Camera` color matrices, and CFA-indexed per-channel black levels to guarantee accurate color reproduction in exported DNGs.
* **Shared Memory Orchestration:** Utilizes `multiprocessing.sharedctypes` to map multi-gigabyte 32-bit float image arrays directly into RAM. This prevents the severe memory overhead of passing large NumPy arrays between standard Python processes.
* **Hardware Safety Gate:** Real-time monitoring of system RAM (via `psutil`) pauses CPU dispatching if free memory falls below the user-defined threshold in `settings.json`.
* **Live Telemetry:** Writes asynchronous output to `training_results.csv`, logging completion times, parameters, and success states for long-term analytics.

---

### Supported File Formats

The engine utilizes `rawpy` (LibRaw) for camera sensor data and `cv2`/`tifffile` for raster and DNG tag data, maintaining a high-precision 16-bit signal chain regardless of the source.

#### Supported Input Formats:

* **Digital Negatives (Bayer CFA & Linear RAW):** `.dng`, `.orf`, `.nef`, `.cr2`, `.arw`, `.rw2`, `.raf`
* **Standard Raster:** `.tif` / `.tiff` (preferred), `.png`, `.jpg`, `.jpeg`, `.bmp` (Processed via RGB Pipeline)

#### Output Formats:

* **TIFF (Default for Standard Images):** 16-bit, uncompressed standard TIFF.
* **DNG (Default for RAW):** 16-bit Uncompressed Digital Negative, preserving reconstructed Bayer data with injected color matrices, per-channel black/white levels, and `AsShotNeutral` white balance tags.

---

### How to Use

The engine is intended for use as a command-line utility. Due to upload limits i cannot distribute a pre packed binary file.

```bash
./IRK_CS-Upscale [file]

```

Process a single image using `settings.json` parameters. The engine automatically selects the `.dng` (RAW) or `.tif` (RGB) output format based on the input type.

```bash
./IRK_CS-Upscale --batch [directory]

```

**Production Mode:** Processes all supported images in a directory. Skips already-upscaled files to allow safe resuming of interrupted jobs, automatically routing files to their respective CFA or RGB processing paths.

```bash
./IRK_CS-Upscale --test [file/dir]

```

**Visual Training Mode:** Generates a 9-way parameter sweep grid to manually find optimal solver weights. RAW inputs output 9 `.dng` files; RGB inputs output 9 `.tif` files. Outputs are saved as labeled files (e.g., `_L1e-05_T0.002.dng` or `.tif`) for direct evaluation.

```bash
./IRK_CS-Upscale --tune [directory] [--full]

```

**Auto-Tuning Engine:** Reads a folder of sample images and automatically calculates the mathematical optimum for Lambda and TV weights, writing the winner directly to `settings.json`. By default, it operates on a fast `512x512` central crop. Append `--full` to bypass the heuristic and tune against the entire image.

```bash
./IRK_CS-Upscale [file] --dng

```

Append `--dng` to any command to explicitly force `.dng` container output.

---

### Configuration

The engine requires a `settings.json` in the execution directory to define hardware and math limits *(note: `MAX_THREADS` are dynamically distributed across 3 channels for RGB/Linear DNG or 4 channels for Bayer CFA)*:

```json
{
    "SCALE_FACTOR": 2,
    "TILE_SIZE": 64,
    "OVERLAP": 16,
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

`CS_Verify` is a companion diagnostic utility designed to mathematically quantify the fidelity of Compressed Sensing reconstructions against downsampled reference targets.

#### Analytical Metrics

The suite utilizes five primary mathematical benchmarks to evaluate signal integrity and solver convergence:

1. **Structural Similarity Index (SSIM)**
Models the human visual system’s perception of luminance, contrast, and structure (Target: $> 0.980$).

$$SSIM(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$


2. **Pearson Correlation ($r$)**
Measures the linear dependence between pixel intensities of the reference and reconstructed images, evaluating structural and textural alignment independent of global luminance or contrast shifts (Target: $> 0.990$).

$$r = \frac{\sum_{i=0}^{m-1} \sum_{j=0}^{n-1} [I(i,j) - \mu_I][K(i,j) - \mu_K]}{\sqrt{\sum_{i=0}^{m-1} \sum_{j=0}^{n-1} [I(i,j) - \mu_I]^2 \sum_{i=0}^{m-1} \sum_{j=0}^{n-1} [K(i,j) - \mu_K]^2}}$$


3. **Peak Signal-to-Noise Ratio (PSNR)**
Quantifies the ratio between the maximum possible signal power and the power of corrupting noise introduced during the 500-iteration solver process (Target: $> 30\text{ dB}$).

$$PSNR = 10 \cdot \log_{10}\left(\frac{MAX_I^2}{MSE}\right)$$


4. **Mean Squared Error (MSE)**
Represents the cumulative squared error between the reconstructed image and the original reference (Target: $\to 0$).

$$MSE = \frac{1}{mn} \sum_{i=0}^{m-1} \sum_{j=0}^{n-1} [I(i,j) - K(i,j)]^2$$


5. **Error Heatmapping**
Generates a spatial divergence map using the Jet Colormap to identify tile boundary artifacts or regions where `TV_WEIGHT` over-smoothed microscopic textures.

---

### Technical Usage (CS_Verify)

#### Requirements

* Python 3.10+
* Dependencies: `opencv-python`, `scikit-image`, `matplotlib`, `scipy`

#### Execution Command

```bash
python cs_verify.py <Original_Path> <Reconstructed_Path>

```

#### Output Artifacts

1. **Console Summary:** A quick-look table of MSE, PSNR, Pearson Correlation, and SSIM.
2. **`Comparison_Report.pdf`:** A multi-page diagnostic report containing:
* **Side-by-Side Visuals:** The reference image vs. the reconstruction.
* **Normalized Difference Map:** Highlighting the exact spatial location of residuals.
* **Interpretation Glossary:** Metric definitions to guide parameter tuning for subsequent batch runs.



    [![Reconstruction Report Preview](Comparison_Report.jpg)](Comparison_Report.jpg)

