import numpy as np
from src import features


def estimate_label(image):
    """
    Takes in RGB image input. Analyzes the image using feature extraction code and output a one-hot encoded label
    """
    masked_image = features.create_mask_filter(image)
    feature = features.create_brightness_feature(masked_image)

    # make sure filter doesn't black out the whole image
    if np.sum(feature) < 100:
        feature = features.create_brightness_feature(image)

    # extract index from feature vector with the highest value
    est_index = np.where(feature == np.amax(feature))[0][0]
    predicted_label = [0, 0, 0]
    predicted_label[est_index] = 1

    return predicted_label


def get_misclassified_images(test_images):
    """
    Constructs a list of misclassified images given a list of tests images and their labels.
    This will throw an AssertionError if labels are not standardized (one-hot encoded)
    """
    # Track misclassified images by placing them into a list
    misclassified_images_labels = []

    # Iterate through all the tests images
    # Classify each image and compare to the true label
    for image in test_images:

        # Get true data
        im = image[0]
        true_label = image[1]
        assert (len(true_label) == 3), "The true_label is not the expected length (3)."

        # Get predicted label from classifier
        predicted_label = estimate_label(im)
        assert (len(predicted_label) == 3), "The predicted_label is not the expected length (3)."

        # Compare true and predicted labels
        if (predicted_label != true_label):
            # If these labels are not equal, the image has been misclassified
            misclassified_images_labels.append((im, predicted_label, true_label))

    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels


def calculate_accuracy(initial_images, misclassified_images):
    """
    Takes in initial image list and misclassified list after classification, calculates and prints out accuracy
    """
    total = len(initial_images)
    misclassified = len(misclassified_images)
    num_correct = total - misclassified
    accuracy = num_correct / total
    print('\n' + '--------------------Begin A Model Classification----------------------------' + '\n')
    print('Accuracy: %s.' % accuracy)
    print("Number of misclassified images = %s out of %s." % (misclassified, total))
    print('\n' + '-----------The model is trained and evaluated successfully!-----------------' + '\n')
