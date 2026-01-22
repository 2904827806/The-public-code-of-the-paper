# Process overexposed areas
import os
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np

def remove_small_black_areas(image_path, output_path, min_area):
    """
    Filter small black noise areas in a black-and-white image.
    Parameters:
    - image_path: Path to input black-and-white image
    - output_path: Path to save output image
    - min_area: Area threshold; black regions smaller than this will be removed
    """
    # Read image (ensure grayscale)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if image is loaded successfully
    if img is None:
        print("Error: Image not found at the specified path.")
        return

    # Invert image so black areas become foreground (255), white becomes background (0)
    inverted = cv2.bitwise_not(img)

    # Connected components analysis
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)

    # Create a new blank image
    filtered_image = np.zeros_like(img)

    # Iterate through connected components and keep those with area >= min_area
    for i in range(1, num_labels):  # Skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            filtered_image[labels == i] = 255  # Keep this component

    # Invert back to original black-and-white format
    result = cv2.bitwise_not(filtered_image)

    # Save result
    cv2.imwrite(output_path, result)

def remove_black_areas(base_image_path, black_image_path, output_path):
    """
    Remove areas from the first image corresponding to black regions in the second image (set to white).

    :param base_image_path: Path to the first image
    :param black_image_path: Path to the second image
    :param output_path: Path to save the output image
    """
    # Read both images
    base_img = cv2.imread(base_image_path)
    black_img = cv2.imread(black_image_path)

    # Ensure images have the same resolution
    if base_img.shape != black_img.shape:
        black_img = cv2.resize(black_img, (base_img.shape[1], base_img.shape[0]))

    # Extract black areas from the second image (B, G, R all 0)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([0, 0, 0])
    black_mask = cv2.inRange(black_img, lower_black, upper_black)

    # Create inverse mask: keep non-black areas
    inverse_mask = cv2.bitwise_not(black_mask)

    # Apply mask to the first image; remove black regions
    base_img[black_mask == 255] = [255, 255, 255]  # Set black regions to white

    # Save result
    cv2.imwrite(output_path, base_img)

def overlay_black_area(base_image_path, overlay_image_path, output_path):
    """
    Overlay black areas from the second image onto the first image, without affecting
    the clarity of the first image or the blackness of overlayed areas.

    :param base_image_path: Path to the first image
    :param overlay_image_path: Path to the second image
    :param output_path: Path to save the output image
    """
    # Read both images
    base_img = cv2.imread(base_image_path)
    overlay_img = cv2.imread(overlay_image_path)

    # Ensure images have the same size
    if base_img.shape != overlay_img.shape:
        overlay_img = cv2.resize(overlay_img, (base_img.shape[1], base_img.shape[0]))

    # Extract black areas from the second image (B, G, R all 0)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([0, 0, 0])
    black_mask = cv2.inRange(overlay_img, lower_black, upper_black)

    # Create inverse mask: keep non-black areas
    inverse_mask = cv2.bitwise_not(black_mask)

    # Overlay black areas onto the first image
    combined = cv2.bitwise_and(base_img, base_img, mask=inverse_mask)  # Keep non-black parts of base image
    black_overlay = cv2.bitwise_and(overlay_img, overlay_img, mask=black_mask)  # Extract black parts from overlay

    # Merge both parts
    result = cv2.add(combined, black_overlay)

    # Save result
    cv2.imwrite(output_path, result)


# Folder containing color moment images
imgs = r""
# Folder containing black-and-white images
fs = r""
# Temporary folder for intermediate processing
ps = r""
# Temporary folder for intermediate processing
qc = r""
# Temporary folder for intermediate processing
op = r""
# Temporary folder for intermediate processing
op1 = r""
# Temporary folder for intermediate processing
op11 = r""
# Output folder for final processed images
op2 = r""


for file in os.listdir(fs):
    if not os.path.exists(ps):
        os.makedirs(ps)
    if not os.path.exists(qc):
        os.makedirs(qc)
    if not os.path.exists(op):
        os.makedirs(op)
    if not os.path.exists(op1):
        os.makedirs(op1)
    if not os.path.exists(op11):
        os.makedirs(op11)
    if not os.path.exists(op2):
        os.makedirs(op2)
    print(file,'Start processing')

    # 1. Remove non-overexposed areas
    i = *  # Adjust according to target
    input_image0 = os.path.join(ps, file)  # Input image path
    print('2. Start removing noise')
    min_area_threshold0 = i  # Set area threshold; black areas smaller than this will be removed
    output_image0 = os.path.join(qc, file)  # Output image path
    remove_small_black_areas(input_image0, output_image0, min_area_threshold0)
    img = cv2.imread(output_image0)
    kernel = np.ones((*, *), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=*)
    output_image01 = os.path.join(op11, file)
    cv2.imwrite(output_image01, erosion)

    # 2. Remove overexposed areas from the original image
    print('3. Remove overexposed areas from original image')
    base_image_path = os.path.join(fs, file)  # First image path
    black_image_path = output_image01  # Second image path
    output_path0 = os.path.join(op, file)  # Output image path
    remove_black_areas(base_image_path, black_image_path, output_path0)

    # 3. Remove small black noise
    i1 = *  # Specify the threshold value
    # Input and output example
    input_image = output_path0  # Input image path
    print('4. Start removing small black noise')
    min_area_threshold = i1  # Set area threshold; black regions smaller than i1 will be removed
    output_image = os.path.join(op1, file)  # Generate output filename
    remove_small_black_areas(input_image, output_image, min_area_threshold)

    # 4. Feature overlay / fusion
    # Example
    print('5. Start overlaying features')
    base_image_path = os.path.join(imgs, file)  # Path to the first image
    overlay_image_path = output_image  # Path to the second image
    output_path = os.path.join(op2, file)  # Path to save output image
    overlay_black_area(base_image_path, overlay_image_path, output_path)

shutil.rmtree(qc)
shutil.rmtree(ps)
#shutil.rmtree(op1)
shutil.rmtree(op11)
shutil.rmtree(op)
