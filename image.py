from functools import lru_cache

import numpy as np
from PIL import Image

def image_similarity(image1_array, image2_array):
    """Compute a stronger image similarity score in [0, 1]."""
    # Ensure the images have the same shape
    if image1_array.shape != image2_array.shape:
        raise ValueError("Images must have the same dimensions for similarity computation.")

    image1 = image1_array.astype(np.float32) / 255.0
    image2 = image2_array.astype(np.float32) / 255.0

    image_1_histogram_red, _ = np.histogram(image1_array[:, :, 0].flatten(), bins=256, range=(0, 256), density=True)
    image_2_histogram_red, _ = np.histogram(image2_array[:, :, 0].flatten(), bins=256, range=(0, 256), density=True)

    image_1_histogram_green, _ = np.histogram(image1_array[:, :, 1].flatten(), bins=256, range=(0, 256), density=True)
    image_2_histogram_green, _ = np.histogram(image2_array[:, :, 1].flatten(), bins=256, range=(0, 256), density=True)

    image_1_histogram_blue, _ = np.histogram(image1_array[:, :, 2].flatten(), bins=256, range=(0, 256), density=True)
    image_2_histogram_blue, _ = np.histogram(image2_array[:, :, 2].flatten(), bins=256, range=(0, 256), density=True)

    histogram_score_red = float(np.sum(np.minimum(image_1_histogram_red, image_2_histogram_red)))
    histogram_score_green = float(np.sum(np.minimum(image_1_histogram_green, image_2_histogram_green)))
    histogram_score_blue = float(np.sum(np.minimum(image_1_histogram_blue, image_2_histogram_blue)))

    histogram_score = float((histogram_score_red + histogram_score_green + histogram_score_blue) / 3.0)

    mae_score = float(1.0 - np.clip(np.mean(np.abs(image1 - image2)), 0.0, 1.0))

    grad_x_1 = image1[:, 1:, :] - image1[:, :-1, :]
    grad_x_2 = image2[:, 1:, :] - image2[:, :-1, :]
    grad_y_1 = image1[1:, :, :] - image1[:-1, :, :]
    grad_y_2 = image2[1:, :, :] - image2[:-1, :, :]
    grad_diff = 0.5 * (np.mean(np.abs(grad_x_1 - grad_x_2)) + np.mean(np.abs(grad_y_1 - grad_y_2)))
    edge_score = float(1.0 - np.clip(grad_diff / 2.0, 0.0, 1.0))

    similarity = (0.2 * histogram_score) + (0.4 * mae_score) + (0.4 * edge_score)
    return float(np.clip(similarity, 0.0, 1.0))


# Standard sRGB to CIE 1931 XYZ transformation matrix (D65 white point, linear values)
SRGB_TO_XYZ_MATRIX = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]
], dtype=np.float64)

@lru_cache(maxsize=128)
def estimate_white_balance(image_path: str) -> int:
    """
    Estimate the Correlated Color Temperature (CCT) in Kelvin using:
      1. Gray World Hypothesis (mean R, G, B intensities)
      2. sRGB -> CIE 1931 XYZ conversion via standard matrix
      3. Chromaticity (x, y) derivation
      4. McCamy's cubic approximation for CCT

    Args:
        image_path (str): Path to an image file.

    Returns:
        int: Estimated CCT in Kelvin, or 6500K (daylight fallback) for errors/edge cases.
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Failed to open image at {image_path}: {exc}") from exc

    arr = np.array(img, dtype=np.float64) / 255.0

    # 1. Gray World Hypothesis: mean intensities per channel
    r_mean = float(np.mean(arr[:, :, 0]))
    g_mean = float(np.mean(arr[:, :, 1]))
    b_mean = float(np.mean(arr[:, :, 2]))
    

    # Robustness: black image (all channels near zero) -> division-by-zero prevention
    if r_mean < 1e-6 and g_mean < 1e-6 and b_mean < 1e-6:
        return 6500

    # Robustness: pure white image
    if r_mean > 0.999 and g_mean > 0.999 and b_mean > 0.999:
        return 6500

    # Normalize by green mean (Gray World assumption)
    g_safe = g_mean if g_mean > 1e-6 else 1e-6
    r_norm = r_mean / g_safe
    g_norm = 1.0
    b_norm = b_mean / g_safe

    # 2. Convert normalized linear RGB to XYZ using standard sRGB matrix
    rgb_vec = np.array([r_norm, g_norm, b_norm], dtype=np.float64)
    xyz_vec = SRGB_TO_XYZ_MATRIX @ rgb_vec
    X, Y, Z = float(xyz_vec[0]), float(xyz_vec[1]), float(xyz_vec[2])

    # 3. Derive chromaticity coordinates (x, y)
    denom = X + Y + Z
    if denom < 1e-9:
        return 6500
    x_chrom = X / denom
    y_chrom = Y / denom

    # 4. McCamy's cubic approximation for CCT
    #    n = (x - 0.3320) / (y - 0.1858)
    y_offset = y_chrom - 0.1858
    if abs(y_offset) < 1e-6:
        return 6500
    n = (x_chrom - 0.3320) / y_offset
    cct_approx = -437 * (n ** 3) + 3601 * (n ** 2) - 6861 * n + 5517

    # Clamp to realistic light-source range and round
    cct_clamped = int(round(np.clip(cct_approx, 2000, 15000)))
    return cct_clamped


if __name__ == "__main__":
    import os, json
    dataset_path = os.path.join("dataset", "photos.json")
    if os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        for entry in dataset:
            input_path = os.path.join("dataset", entry.get("input", ""))
            if os.path.isfile(input_path):
                try:
                    temp = estimate_white_balance(input_path)
                    print(f"Estimated white balance for {input_path}: {temp} K")
                except Exception as exc:
                    print(f"Error processing {input_path}: {exc}")
    else:
        print("Example dataset not found at dataset/photos.json")
