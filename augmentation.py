import albumentations as A
import cv2  
import numpy as np

def read_image(image_path):
    """
    Read an image from a file path and convert it to a numpy array.

    Args:
        image_path (str): The path to the image file.

    Returns:
        numpy.ndarray: The image as a numpy array.
    """
    # Read the image using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Preserve alpha
    
    # Convert BGRA to RGBA if alpha is present
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image
    
    return image
def augment_image(image):
    """
    Apply a series of augmentations to the input image.

    Args:
        image (numpy.ndarray): The input image to be augmented.

    Returns:
        numpy.ndarray: The augmented image.
    """
    height, width = image.shape[:2]

    # Calculate the diagonal length of the image to determine the required canvas size
    diagonal = int(np.sqrt(height**2 + width**2))
    padded_height = diagonal
    padded_width = diagonal

    # Pad the image to the new canvas size
    transform_pad = A.PadIfNeeded(
        min_height=padded_height,
        min_width=padded_width,
        border_mode=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)  # Black padding
    )

    # Define the augmentation pipeline
    transform = A.Compose([
        transform_pad,
        A.Rotate(limit=(0, 360), border_mode=cv2.BORDER_CONSTANT, p=1.0),  # Full rotation
    ])

    # Apply the augmentations
    augmented = transform(image=image)
    
    return augmented['image']

#print(read_image("data_for_augmentation/1.png"))

def place_on_background(augmented_image, background_image):
    """
    Place the augmented image (with transparency) onto a background image at random coordinates.

    Args:
        augmented_image (numpy.ndarray): The augmented image to be placed (with alpha channel).
        background_image (numpy.ndarray): The background image.

    Returns:
        numpy.ndarray: The combined image with the augmented image placed on the background.
    """
    # Ensure the augmented image has an alpha channel
    if augmented_image.shape[2] != 4:
        raise ValueError("The augmented image must have an alpha channel (RGBA).")

    # Ensure the background image is RGB
    if background_image.shape[2] == 4:  # If background has an alpha channel, remove it
        background_image = cv2.cvtColor(background_image, cv2.COLOR_RGBA2RGB)

    bg_height, bg_width = background_image.shape[:2]
    aug_height, aug_width = augmented_image.shape[:2]

    # Ensure the augmented image fits within the background
    if aug_height > bg_height or aug_width > bg_width:
        raise ValueError("The augmented image is larger than the background image.")

    # Generate random coordinates for placing the augmented image
    max_x = bg_width - aug_width
    max_y = bg_height - aug_height
    x_offset = np.random.randint(0, max_x + 1)
    y_offset = np.random.randint(0, max_y + 1)

    # Split the augmented image into RGB and alpha channels
    aug_rgb = augmented_image[:, :, :3]
    aug_alpha = augmented_image[:, :, 3] / 255.0  # Normalize alpha to range [0, 1]

    # Extract the region of interest (ROI) from the background
    roi = background_image[y_offset:y_offset + aug_height, x_offset:x_offset + aug_width]

    # Blend the augmented image with the ROI using the alpha channel
    blended = (aug_rgb * aug_alpha[:, :, None] + roi * (1 - aug_alpha[:, :, None])).astype(np.uint8)

    # Place the blended region back onto the background
    combined_image = background_image.copy()
    combined_image[y_offset:y_offset + aug_height, x_offset:x_offset + aug_width] = blended

    return combined_image

def save_image(image, output_path):
    """
    Save the augmented image to a file.

    Args:
        image (numpy.ndarray): The image to be saved.
        output_path (str): The path where the image will be saved.
    """
    # Convert the image from RGB to BGR format
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_path, image)


image = read_image("data_for_augmentation/driver02.png")
background_image = read_image("backgrounds\pexels-andrejcook-131723.jpg")
transformed_image = augment_image(image)
combined = place_on_background(transformed_image, background_image)
save_image(transformed_image, 'test/transformed_image1.png')
save_image(combined, 'test/test_background1.png')