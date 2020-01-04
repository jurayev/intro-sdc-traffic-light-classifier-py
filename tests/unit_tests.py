import unittest
from src import preprocess


def print_fail():
    print("TEST FAILED: see details below.")


def print_pass(message):
    print("TEST PASSED: %s" % message)


def run_unittest(misclassified_images):
    print("--------------------------RUNNING UNIT TESTS--------------------------------")
    tests = Tests()
    tests.test_one_hot_encode_red()
    tests.test_one_hot_encode_yellow()
    tests.test_one_hot_encode_green()
    tests.test_red_as_green(misclassified_images)


# A class holding all tests
class Tests(unittest.TestCase):

    def test_one_hot_encode_red(self):
        expected_label = [1, 0, 0]
        encoded_label = preprocess.one_hot_encode('red')
        self.assertListEqual(encoded_label, expected_label,
                             "one_hot_encode() works incorrectly. Expected %s. Actual: %s" % (
                                 expected_label, encoded_label))
        print_pass("one_hot_encode() works as expected for RED images!")

    def test_one_hot_encode_yellow(self):
        expected_label = [0, 1, 0]
        encoded_label = preprocess.one_hot_encode('yellow')
        self.assertListEqual(encoded_label, expected_label,
                             "one_hot_encode() works incorrectly. Expected %s. Actual: %s" % (
                                 expected_label, encoded_label))
        print_pass("one_hot_encode() works as expected for YELLOW images!")

    def test_one_hot_encode_green(self):
        expected_label = [0, 0, 1]
        encoded_label = preprocess.one_hot_encode('green')
        self.assertListEqual(encoded_label, expected_label,
                             "one_hot_encode() works incorrectly. Expected %s. Actual: %s" % (
                                 expected_label, encoded_label))
        print_pass("one_hot_encode() works as expected for GREEN images!")

    def test_red_as_green(self, misclassified_images):
        """
        Tests if any misclassified images are red but mistakenly classified as green
        """
        # Loop through each misclassified image and the labels
        for im, predicted_label, true_label in misclassified_images:

            # Check if the image is one of a red light
            if true_label == [1, 0, 0]:
                try:
                    # Check that it is NOT labeled as a green light
                    self.assertNotEqual(predicted_label, [0, 0, 1])
                except self.failureException as e:
                    print_fail()
                    print("Warning: A red light is classified as green.")
                    print('\n' + str(e))
                    return
        print_pass("test_red_as_green() - No misclassified Red images are classified as Green!")
