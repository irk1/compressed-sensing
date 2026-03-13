

### CSNEW-Image_Gen: High-Precision Compressed Sensing Upscaler

CSNEW-Image_Gen is a professional-grade image reconstruction engine designed for high-bit-depth digital negatives. Unlike standard AI upscalers that generate pixels via neural networks, this tool uses Compressed Sensing and Basis Pursuit Denoising to reconstruct signals using mathematical optimization.

The system is optimized for Dual-Xeon Workstations, leveraging high core counts and massive RAM pools to solve complex convex optimization problems at 16-bit precision.

---

### Basis of Function

The core of the program is based on the premise that natural images are sparse in a transform domain, specifically the Discrete Cosine Transform.

1. Mathematical Objective
The solver reconstructs an image x by minimizing a multi-objective cost function.


$$\min_{x} \lambda | \Psi x |_1 + \gamma \text{TV}(x)$$

Mathematica:
Minimize[lambda * Total[Abs[DCT[x]]] + gamma * TotalVariation[x], constraints]

Subject to the constraint:

$$| Ax - y |_2 \leq \epsilon$$

Mathematica:
Norm[A . x - y, 2] <= epsilon

* $\Psi$ : The DCT transform operator.
* $L1 \ Norm$: Promotes sparsity, ensuring only the most significant image structures are preserved.
* $TV(x)$: Total Variation regularization to reduce noise while preserving sharp edges.
* $A$: The downsampling operator representing the relationship between the desired high-res image and the low-res sensor data $y$.

2. The 32-Bit Float Pipeline
To avoid the precision loss inherent in 8-bit or 16-bit integer processing, the engine operates in 32-bit Floating Point space. This allows the ECOS solver to converge on solutions with an absolute tolerance of 10^-8, preserving the full dynamic range of RAW sensor data.

---

### Development and Architecture

* Core Logic: Python 3.10+
* Optimization Solver: CVXPY with the ECOS (Embedded Conic Solver) backend.
* Image IO: rawpy (LibRaw) for DNG/RAW decoding and OpenCV for color space transformations.
* Parallelization: ProcessPoolExecutor utilizing a Hardware Safety Gate to monitor system RAM before submitting new optimization tiles.

---

### How to Use

Command Line Interface

./CSNEW-Image_Gen [file]
Process a single image using settings.json parameters.

./CSNEW-Image_Gen --batch [directory]
Production mode: Processes all images in a directory.

./CSNEW-Image_Gen --test [file/dir]
Training Mode: Generates a 9-way parameter sweep for tuning.

---

### Analytics and Scoring

The program includes an analytical suite to quantify reconstruction quality rather than relying on subjective observation.

1. Parameter Sweeping (The Training Portion)
When running in --test mode, the program generates a grid based on variable lambda (Sparsity) and gamma (TV Weight).

* Sparsity Sweep: Tests values $10^-6$, $10^-5$, $10^-4$ to determine the sensor Texture-to-Artifact breakpoint.
* TV Sweep: Tests levels of edge-preserving smoothing 0.001, 0.002, 0.005.

2. Scoring Metrics
The engine calculates fidelity using two primary metrics:

* SSIM (Structural Similarity Index): Measures the degradation of structural information between the original sensor data and the reconstructed result.
Target range: 0.975 to 0.992.
* Convergence Residuals: The ECOS solver provides a residual score. If the solver fails to reach the specified tolerance within 5,000 iterations, the program identifies the tile as Low Confidence and applies a high-fidelity Lanczos4 fallback.

---

### Hardware Optimization for Xeon Servers

The engine is specifically tuned for high-core-count environments:

* Thread Allocation: Defaulted to 40 threads to saturate dual-socket configurations.
* Memory Management: The script monitors psutil.virtual_memory() and pauses if the MAX_RAM_GB threshold is exceeded.