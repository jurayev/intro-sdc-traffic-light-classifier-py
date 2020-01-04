import random
from src import helpers, model, preprocess
from tests import unit_tests


def main():
    # Image data directories
    IMAGE_DIR_TRAINING = "traffic_light_images/training/"

    # Using the load_dataset function in helpers.py
    # Load tests data
    TEST_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)

    # Standardize the tests data
    STANDARDIZED_TEST_LIST = preprocess.standardize(TEST_IMAGE_LIST)

    # Shuffle the standardized tests data
    random.shuffle(STANDARDIZED_TEST_LIST)

    # Find all misclassified images in a given tests set
    MISCLASSIFIED = model.get_misclassified_images(STANDARDIZED_TEST_LIST)

    model.calculate_accuracy(STANDARDIZED_TEST_LIST, MISCLASSIFIED)

    # TESTS
    unit_tests.run_unittest(MISCLASSIFIED)


if __name__ == '__main__':
    main()
