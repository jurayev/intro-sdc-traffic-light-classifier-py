import cv2
import numpy as np


def create_brightness_feature(image):
    """
    Creates a brightness feature that takes in an RGB image and returns a feature vector with HSV colorspace values
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2]
    # Red, Yellow, Green boundary
    boundary = int(len(v) / 3)

    feature = [0, 0, 0]
    feature[0] = np.sum(v[:boundary])
    feature[1] = np.sum(v[boundary:-boundary])
    feature[2] = np.sum(v[-boundary:])

    return feature


def create_mask_filter(rgb_image):
    """
    Creates a masked image representation that takes in an RGB image and returns a new color masked version
    """
    # Copy an image before modifying
    image = np.copy(rgb_image)

    # First mask dark colors if any excluding Red, Yellow, Green
    lower_dark_color = np.array([0, 0, 0])
    upper_dark_color = np.array([114, 125, 120])
    dark_mask = cv2.inRange(image, lower_dark_color, upper_dark_color)
    image[dark_mask != 0] = [0, 0, 0]

    # Then mask bright colors if any excluding Red, Yellow, Green
    lower_bright_color = np.array([100, 100, 100])
    upper_bright_color = np.array([230, 230, 225])
    bright_mask = cv2.inRange(image, lower_bright_color, upper_bright_color)
    image[bright_mask != 0] = [0, 0, 0]

    return image
