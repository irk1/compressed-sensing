import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)

import numpy as np
import pywt
from skimage import io, img_as_float, img_as_ubyte
from scipy.optimize import minimize
import tifffile

def wavelet_reconstruct(coeffs, wavelet, mode='periodization'):
    return pywt.waverec2(coeffs, wavelet, mode=mode)

def wavelet_decompose(image, wavelet, level=2, mode='periodization'):
    return pywt.wavedec2(image, wavelet, level=level, mode=mode)

def sparse_reconstruct(measurements, mask, wavelet, shape, level=2):
    def loss_function(coeffs_flat):
        coeffs = pywt.array_to_coeffs(coeffs_flat, coeff_slices, output_format='wavedec2')
        reconstructed = wavelet_reconstruct(coeffs, wavelet)
        return np.linalg.norm((reconstructed * mask - measurements), 2)

    zero_image = np.zeros(shape)
    coeffs = wavelet_decompose(zero_image, wavelet, level)
    coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)

    result = minimize(loss_function, coeff_arr, method='L-BFGS-B', options={'maxiter': 100})
    reconstructed_coeffs = pywt.array_to_coeffs(result.x, coeff_slices, output_format='wavedec2')
    return wavelet_reconstruct(reconstructed_coeffs, wavelet)

def interpolate_channel(channel, wavelet='db1', level=2, scale=2):
    h, w = channel.shape
    hr_shape = (h * scale, w * scale)

    upsampled = np.zeros(hr_shape)
    mask = np.zeros(hr_shape)

    upsampled[::scale, ::scale] = channel
    mask[::scale, ::scale] = 1

    reconstructed = sparse_reconstruct(upsampled, mask, wavelet, hr_shape, level)
    return np.clip(reconstructed, 0, 1)

def process_rgb_image(input_path, output_path, scale=2, wavelet='db1', level=2):
    img = io.imread(input_path)
    img = img_as_float(img)

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Input image must be an RGB image.")

    print("Starting RGB compressed sensing upscaling...")

    channels = []
    for i, color in enumerate(['Red', 'Green', 'Blue']):
        print(f"Processing {color} channel...")
        channel = img[:, :, i]
        upscaled = interpolate_channel(channel, wavelet=wavelet, level=level, scale=scale)
        channels.append(upscaled)

    img_hr = np.stack(channels, axis=2)
    img_hr = img_as_ubyte(np.clip(img_hr, 0, 1))

    print("Saving high-resolution TIFF image...")
    tifffile.imwrite(output_path, img_hr)
    print(f"Done! Saved to {output_path}")

# Example usage
if __name__ == "__main__":
    input_image_path = "input_image.jpg"
    output_tiff_path = "output_image_rgb.tiff"
    process_rgb_image(input_image_path, output_tiff_path)
