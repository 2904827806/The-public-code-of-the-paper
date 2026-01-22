import numpy as np
import rasterio
from rasterio.transform import from_origin
import os

# Read multispectral image
def read_multispectral_image(image_path):
    with rasterio.open(image_path) as src:
        img = src.read()  # Read all bands
        transform = src.transform  # Get geotransform
        crs = src.crs  # Get coordinate reference system
        height, width = img.shape[1], img.shape[2]  # Get image height and width
        return img, transform, crs, height, width


# Calculate color moment features
def color_moments(img, height, width):
    n_bands = img.shape[0]  # Get number of bands
    all_color_feature = np.zeros((n_bands, 3))  # Compute 3 features per band: mean, std, third moment
    color_features = np.zeros((3, height, width))  # Save 3 features per pixel

    for i in range(n_bands):
        band = img[i]  # Get current band data

        # Calculate first moment (mean)
        mean = np.mean(band)

        # Calculate second moment (standard deviation)
        std = np.std(band)

        # Calculate third moment (cube root of skewness)
        skewness = np.mean(abs(band - np.mean(band)) ** 3)
        third_moment = skewness ** (1. / 3)

        # Save 3 features for each band
        all_color_feature[i] = [mean, std, third_moment]

        # Assign band color moments to feature matrix
        color_features[0] += band  # Sum of means of all bands
        color_features[1] += (band - mean) ** 2  # Sum of variances of all bands
        color_features[2] += abs(band - mean) ** 3  # Sum of third moments

    # Calculate standard deviation and third moment
    color_features[1] = np.sqrt(color_features[1] / n_bands)  # Root mean square std
    color_features[2] = (color_features[2] / n_bands) ** (1 / 3)  # Third moment

    return color_features


# Save feature matrix as raster
def save_color_moments_as_raster(features, output_path, transform, crs, height, width):
    n_features = features.shape[0]
    with rasterio.open(
            output_path, 'w', driver='GTiff', count=n_features, dtype='float32',
            width=width, height=height, crs=crs, transform=transform
    ) as dst:
        for i in range(n_features):
            dst.write(features[i], i + 1)


# Main function
def main(input_image_path, output_image_path):
    # Read remote sensing image
    img, transform, crs, height, width = read_multispectral_image(input_image_path)

    # Calculate color moment features
    color_features = color_moments(img, height, width)

    # Save feature matrix as raster
    save_color_moments_as_raster(color_features, output_image_path, transform, crs, height, width)


# Input and output paths
files = r""
oputs = r""
for file in os.listdir(files):
    if file.endswith(".tif"):
        input_image_path = os.path.join(files, file)
        output_image_path = os.path.join(oputs, file)
        print('Start extracting color features for {}'.format(file))
        main(input_image_path, output_image_path)
