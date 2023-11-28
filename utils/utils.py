"""
This module provides utility functions for image processing including
loading settings from a JSON file and applying various image processing
techniques like flipping, enhancing brightness and contrast, and more.
"""
import os
import json
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2


def load_json_settings(file_path):
    """
    Load settings from a JSON file.

    Args:
    file_path (str): Path to the JSON file.

    Returns:
    dict: A dictionary containing settings or None if an error occurs.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e: # pylint: disable=broad-except
        print(f"An error occurred while loading the JSON file: {e}")
        return None


def flip(img):
    """
    Flip the image left to right.

    Args:
    image (Image): An instance of PIL Image.

    Returns:
    Image: The flipped image.
    """
    return img.transpose(Image.FLIP_LEFT_RIGHT) # pylint: disable=no-member


def enhance_brightness(img, i):
    """
    Enhance the brightness of an image.

    Args:
    image (Image): An instance of PIL Image.
    intensity (float): Intensity factor for brightness enhancement.

    Returns:
    Image: The brightness-enhanced image.
    """
    enhancer = ImageEnhance.Brightness(img)
    img_bright = enhancer.enhance(i * 0.5)
    return img_bright


def enhance_contrast(img, i):
    """
    Enhance the contrast of an image.

    Args:
    image (Image): An instance of PIL Image.
    intensity (float): Intensity factor for contrast enhancement.

    Returns:
    Image: The contrast-enhanced image.
    """
    enhancer = ImageEnhance.Contrast(img)
    img_contrast = enhancer.enhance(i * 0.5)
    return img_contrast


def equalize_hist(img):
    """
    Apply histogram equalization to the image.

    Args:
    image (Image): An instance of PIL Image.

    Returns:
    Image: The histogram equalized image.
    """
    return ImageOps.equalize(img)


def sharpen(img):
    """
    Sharpen the image.

    Args:
    image (Image): An instance of PIL Image.

    Returns:
    Image: The sharpened image.
    """
    return img.filter(ImageFilter.SHARPEN)


def edge_enhance(img):
    """
    Enhance the edges of the image.

    Args:
    image (Image): An instance of PIL Image.

    Returns:
    Image: The edge-enhanced image.
    """
    return img.filter(ImageFilter.EDGE_ENHANCE)


def gamma_correction(img, gamma=1.0):
    """
    Apply gamma correction to the image.

    Args:
    image (Image): An instance of PIL Image.
    gamma (float): Gamma value for correction.

    Returns:
    Image: The gamma-corrected image.
    """
    inv_gamma  = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return Image.fromarray(cv2.LUT(np.array(img), table)) # pylint: disable=no-member


def preprocess_and_augment_image(image_path, output_path, **kwargs):
    """
    Process and augment an image based on given parameters.

    Args:
    image_path (str): Path to the input image.
    output_path (str): Path to save the processed image.
    kwargs: Various processing options as keyword arguments.
    """
    original_img = Image.open(image_path)

    if kwargs.get("save_original", False):
        original_img.save(output_path)

    if kwargs.get("flip_mode", False):
        img_flipped = flip(original_img.copy())
        img_flipped.save(os.path.splitext(output_path)[0] + '_flipped.png')

    if kwargs.get("enhance_brightness_mode", False):
        for i in range(3, 4):
            bright_img = enhance_brightness(original_img.copy(), i)
            bright_img.save(os.path.splitext(output_path)[0] + f'_bright_{i}.png')

    if kwargs.get("enhance_contrast_mode", False):
        for i in range(3, 4):
            contrast_img = enhance_contrast(original_img.copy(), i)
            contrast_img.save(os.path.splitext(output_path)[0] + f'_contrast_{i}.png')

    if kwargs.get("sharpen_image_mode", False):
        sharpened_img = sharpen(original_img.copy())
        sharpened_img.save(os.path.splitext(output_path)[0] + '_sharpened.png')

    if kwargs.get("edge_enhance_mode", False):
        edge_enhanced_img = edge_enhance(original_img.copy())
        edge_enhanced_img.save(os.path.splitext(output_path)[0] + '_edge_enhanced.png')

    if kwargs.get("gamma_correct_mode", False):
        gamma_corrected_img = gamma_correction(original_img.copy(), 1.2)
        gamma_corrected_img.save(os.path.splitext(output_path)[0] + '_gamma_corrected.png')

    if kwargs.get("equalize_mode", False):
        equalized_img = equalize_hist(original_img.copy())
        equalized_img.save(os.path.splitext(output_path)[0] + '_equalized.png')
