### CSNEW-Image_Gen: High-Precision Compressed Sensing Upscaler

CSNEW-Image_Gen is a professional-grade image reconstruction engine designed for high-bit-depth digital negatives. Unlike standard AI upscalers that generate pixels via neural networks, this tool uses Compressed Sensing and Basis Pursuit Denoising to reconstruct signals using mathematical optimization.

The system is optimized for Dual-Xeon Workstations, leveraging high core counts and massive RAM pools to solve complex convex optimization problems at 16-bit precision.

---

### Basis of Function

The core of the program is based on the premise that natural images are sparse in a transform domain, specifically the Discrete Cosine Transform (DCT).

#### 1. Mathematical Objective
The solver reconstructs an image $x$ by minimizing a multi-objective cost function:

$$\min_{x} \lambda \| \Psi x \|_1 + \gamma \text{TV}(x)$$

Mathematica:
`Minimize[lambda * Total[Abs[DCT[x]]] + gamma * TotalVariation[x], constraints]`

Subject to the constraint:
$$\| Ax - y \|_2 \leq \epsilon$$

Mathematica:
`Norm[A . x - y, 2] <= epsilon`

* $\Psi$: The DCT transform operator.
* $\| \cdot \|_1$: $L_1$ Norm; promotes sparsity, ensuring only the most significant image structures are preserved.
* $\text{TV}(x)$: Total Variation regularization to reduce noise while preserving sharp edges.
* $A$: The downsampling operator representing the relationship between the desired high-res image and the low-res sensor data $y$.

#### 2. The 32-Bit Float Pipeline
To avoid the precision loss inherent in 8-bit or 16-bit integer processing, the engine operates in 32-bit Floating Point space. This allows the ECOS solver to converge on solutions with an absolute tolerance of $10^{-8}$, preserving the full dynamic range of RAW sensor data.

---

### Architecture & Portability

CSNEW-Image_Gen is designed for Zero-Footprint Deployment on air-gapped servers.

* Self-Contained Binary: Statically bundled using PyInstaller. The executable contains the Python interpreter, ECOS solvers, and all required math libraries.
### 
* Embedded ExifTool: Includes a portable Perl-based ExifTool environment. It automatically copies ColorMatrix, AsShotWhiteValue, and ForwardMatrix from the source RAW to the upscaled DNG.
### 
* Photogrammetry Scaling: Mathematically scales FocalPlaneXResolution and FocalPlaneYResolution based on the SCALE_FACTOR to ensure 1:1 physical sensor accuracy in 3D reconstruction suites.
### 
* Hardware Safety Gate: Real-time monitoring of system RAM (via psutil) pauses processing if free memory falls below the user-defined threshold in settings.json.
### 

---

### Supported File Formats

The engine utilizes `rawpy` and `numpy2dng` to maintain a high-precision 16-bit signal chain from sensor to final export.

#### Supported Input Formats:
* Digital Negatives (RAW): `.dng`, `.nef`, `.cr2`, `.arw`, `.orf`.

* Standard Raster: `.tif` / `.tiff` (preferred), `.png`, `.jpg`.


#### Output: 
* Always exports to 16-bit Uncompressed Digital Negative (`.dng`). This preserves reconstruction data in a format compatible with Adobe Camera Raw, while maintaining original camera color matrices and white balance via internal metadata injection.

---

### How to Use

The engine is distributed as a standalone Linux binary—no local Python or library installation is required on the server.

`./CSNEW-Image_Gen [file]
`
Process a single image using settings.json parameters.

`./CSNEW-Image_Gen --batch [directory]
`
Production mode: Processes all images in a directory.

`./CSNEW-Image_Gen --test [file/dir]
`
Training Mode: Generates a 9-way parameter sweep grid to find the optimal λ and γ. Outputs are saved as labeled DNGs (e.g., `_L1e-05_T0.002.dng`) for direct evaluation in Adobe Camera Raw.

---
### Configuration
The engine requires a `settings.json` in the execution directory to define hardware and math limits:

```
{
    "SCALE_FACTOR": 2,
    "TILE_SIZE": 64,
    "OVERLAP": 12,
    "LAMBDA_MIN": 1e-05,
    "TV_WEIGHT": 0.002,
    "MAX_THREADS": 40,
    "MAX_RAM_GB": 30
}
```



---
### Analytics and Scoring

The program includes an analytical suite to quantify reconstruction quality rather than relying on subjective observation.

#### 1. Parameter Sweeping (Training Mode)
When running in `--test` mode, the program generates a grid based on variable $\lambda$ and $\gamma$.

* Sparsity Sweep: Tests values $10^{-6}$, $10^{-5}$, $10^{-4}$ to determine the sensor Texture-to-Artifact breakpoint.
### 

* TV Sweep: Tests levels of edge-preserving smoothing $0.001$, $0.002$, $0.005$.

#### 2. Scoring Metrics
The engine calculates fidelity using two primary metrics:

* SSIM (Structural Similarity Index): Measures the degradation of structural information between the original sensor data and the reconstructed result. Target range: $0.975$ to $0.992$.  
###  
* Convergence Residuals: If the ECOS solver fails to reach the specified tolerance within 5,000 iterations, the program identifies the tile as Low Confidence and applies a high-fidelity Lanczos4 fallback.  

---




---

### CS_Verify: Reconstruction Fidelity Analytics Suite

The CS_Verify utility is a high-precision diagnostic tool designed to mathematically quantify the success of Compressed Sensing reconstructions. It performs a pixel-level statistical audit between the original sensor data and the final output.

### Analytical Metrics

The suite utilizes four primary mathematical benchmarks to evaluate signal integrity and solver convergence.

#### 1. Structural Similarity Index (SSIM)
Unlike simple error measurements, SSIM models the human visual system’s perception of luminance, contrast, and structure.

* Target: $> 0.980$
* Formula:
$$SSIM(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$

#### 2. Peak Signal-to-Noise Ratio (PSNR)
Quantifies the ratio between the maximum possible power of the signal and the power of corrupting noise introduced during the 5,000-iteration solver process.

* Target: $> 30 \text{ dB}$
* Formula:
$$PSNR = 10 \cdot \log_{10}\left(\frac{MAX_I^2}{MSE}\right)$$

#### 3. Mean Squared Error (MSE)
Represents the cumulative squared error between the reconstructed image and the original reference.

* Target: $\to 0$
* Formula:
$$MSE = \frac{1}{mn} \sum_{i=0}^{m-1} \sum_{j=0}^{n-1} [I(i,j) - K(i,j)]^2$$

#### 4. Error Heatmapping
The program generates a spatial divergence map using the Jet Colormap. This allows for the identification of specific "Tile Boundary" artifacts or areas where the $\gamma$ may have over-smoothed microscopic botanical textures.

---

### Technical Usage

#### Requirements
* Python 3.10+
* Dependencies: `opencv-python`, `skimage`, `matplotlib`, `scipy`

#### Execution Command
`python cs_verify.py <Original_Path> <Reconstructed_Path>`

#### Output Artifacts
1. Console Summary: A quick-look table of MSE, PSNR, Correlation, and SSIM.
2. Comparison_Report.pdf: A professional-grade, multi-page document containing:
    * Side-by-Side Visuals: The reference image vs. the reconstruction.
    * Normalized Difference Map: Highlighting the exact spatial location of residuals.
    * Interpretation Glossary: Definitions for each metric to assist in parameter tuning for the next batch run.

---

### Hardware Integration
This script is designed to run post-batch. It utilizes Lanczos4 Interpolation to standardize dimensions if the reconstruction scale differs from the original, ensuring the statistical correlation is calculated on a true $1:1$ pixel grid. This is critical for 16-bit workflow integrity when comparing upscaled results to downsampled sensor data.




[![Reconstruction Report Preview](Comparison_Report.jpg)](Comparison_Report.jpg)
