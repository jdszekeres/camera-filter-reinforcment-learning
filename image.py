import numpy as np
from PIL import Image

def image_similarity(image1_array, image2_array):
    """Compute a stronger image similarity score in [0, 1]."""
    # Ensure the images have the same shape
    if image1_array.shape != image2_array.shape:
        raise ValueError("Images must have the same dimensions for similarity computation.")

    image1 = image1_array.astype(np.float32) / 255.0
    image2 = image2_array.astype(np.float32) / 255.0

    image_1_histogram, _ = np.histogram(image1_array.flatten(), bins=256, range=(0, 256), density=True)
    image_2_histogram, _ = np.histogram(image2_array.flatten(), bins=256, range=(0, 256), density=True)
    histogram_score = float(np.sum(np.minimum(image_1_histogram, image_2_histogram)))

    mae_score = float(1.0 - np.clip(np.mean(np.abs(image1 - image2)), 0.0, 1.0))

    grad_x_1 = image1[:, 1:, :] - image1[:, :-1, :]
    grad_x_2 = image2[:, 1:, :] - image2[:, :-1, :]
    grad_y_1 = image1[1:, :, :] - image1[:-1, :, :]
    grad_y_2 = image2[1:, :, :] - image2[:-1, :, :]
    grad_diff = 0.5 * (np.mean(np.abs(grad_x_1 - grad_x_2)) + np.mean(np.abs(grad_y_1 - grad_y_2)))
    edge_score = float(1.0 - np.clip(grad_diff / 2.0, 0.0, 1.0))

    similarity = (0.2 * histogram_score) + (0.4 * mae_score) + (0.4 * edge_score)
    return float(np.clip(similarity, 0.0, 1.0))
    
if __name__ == "__main__":
    
    import json
    import os

    os.chdir("dataset")
    dataset = json.load(open("photos.json"))
    for image in dataset:
        image_pretrain = Image.open(image["input"])
        image_pretrain.save("temp_in.jpg")
        import ACR_mockup
        from type import ACRModel
        ACR_mockup.apply_acr_to_pp3(acr=ACRModel(
            exposure=+1,
            dehaze=100,
        ), output_pp3="temp.pp3")
        ACR_mockup.render("temp_in.jpg", "temp.pp3", "temp_out.jpg")
        image1 = Image.open("temp_out.jpg")

        image2 = Image.open(image["output"])  # Replace with the path to your second image

        # Convert images to numpy arrays
        image1_array = np.array(image1)
        image2_array = np.array(image2)

        similarity = image_similarity(image1_array, image2_array)
        print(f"Similarity between the two images: {similarity}")
        