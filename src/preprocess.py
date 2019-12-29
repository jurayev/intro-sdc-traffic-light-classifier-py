import cv2


def standardize_input(image):
    """
    Takes in an RGB image and return a new, standardized version
    """
    resized_version = cv2.resize(image, (32, 32))
    cropped_version = resized_version[7:-4, 10:-10]
    return cropped_version


def one_hot_encode(label):
    """
    Given a label - "red", "green", or "yellow". Returns a one-hot encoded label
    """
    if label == "red":
        return [1, 0, 0]
    if label == "green":
        return [0, 0, 1]
    return [0, 1, 0]


def standardize(image_list):
    """
    Takes in a list of image-label pairs and outputs a standardized list of resized images and one-hot encoded labels
    """
    standard_list = []

    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]
        standardized_im = standardize_input(image)
        one_hot_label = one_hot_encode(label)
        standard_list.append((standardized_im, one_hot_label))

    return standard_list
