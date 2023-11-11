"""
This module provides functionality to process a set of images
using various augmentation techniques.
"""
import os
from tqdm import tqdm
from settings import config

from utils.utils import load_json_settings, preprocess_and_augment_image


def main(parameters, input_dir, output_dir):
    """
    Main function to process images in a given input directory
    and save them in an output directory using specified parameters.

    Args:
    parameters (dict): A dictionary of processing parameters.
    input_dir (str): Path to the directory containing input images.
    output_dir (str): Path to the directory where processed images will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith('.jpg') or f.endswith('.png')]
    total_files = len(files)

    for filename in tqdm(files, total=total_files, desc="Processing images"):
        preprocess_and_augment_image(
            os.path.join(input_dir, filename),
            os.path.join(output_dir, filename),
            **parameters
        )

    print('Number of images processed:', len(files))


if __name__ == '__main__':
    params = load_json_settings(config.APP_PATH_PARAMS_JSON_FILE)
    main(params, config.APP_PATH_INPUT, config.APP_PATH_OUTPUT)
