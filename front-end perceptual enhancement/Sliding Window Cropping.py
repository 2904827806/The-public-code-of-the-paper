from PIL import Image
import os


def crop_image(input_image, output_dir, sizes=256, steps=51, name=''):
    """
    Crop an image into multiple sub-images and save them to the specified directory.
    Parameters:
    - input_image: Path to the input image
    - output_dir: Directory to save the cropped images
    - sizes: Size of each sub-image (default 256x256)
    - steps: Sliding step for cropping (default 51)
    - name: Base name for output images
    """
    # Open the image
    img = Image.open(input_image)
    # Get original image width and height
    width, height = img.size
    print(f"Image size: width={width}, height={height}")

    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Calculate number of columns and rows
    col_num = (width - sizes) // steps + 1
    row_num = (height - sizes) // steps + 1

    # Add extra column/row if remainder exists
    if (width - sizes) % steps != 0:
        col_num += 1
    if (height - sizes) % steps != 0:
        row_num += 1

    # Crop sub-images and save
    num = 1  # Sub-image counter
    for i in range(row_num):
        for j in range(col_num):
            # Compute crop box coordinates
            left = j * steps
            upper = i * steps
            right = min(left + sizes, width)
            lower = min(upper + sizes, height)

            print(f"Crop box: left={left}, upper={upper}, right={right}, lower={lower}")

            # Check crop box validity
            if right <= left or lower <= upper:
                print("Invalid crop box, skipping...")
                continue

            # Crop the image
            cropped_img = img.crop((left, upper, right, lower))

            # Generate filename for cropped image
            output_image = os.path.join(output_dir, f"{name}_{num:04d}_{left}_{upper}.png")

            # Save cropped image
            cropped_img.save(output_image)
            print(f"Saved cropped image: {output_image}")

            num += 1


# Example usage
f = r''  # Input folder containing images
for i in os.listdir(f):
    input_image = os.path.join(f, i)  # Input image path
    output_dir = r''  # Output directory path
    # Parameters
    size = 256  # Crop size
    step = 218  # Step size
    name = i.split('.')[0]
    # Call cropping function
    crop_image(input_image, output_dir, sizes=size, steps=step, name=name)
