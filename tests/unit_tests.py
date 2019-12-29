import unittest


def print_fail():
    print("TEST FAILED")


def print_pass(message):
    print("TEST PASSED")
    print(message)


# A class holding all tests
class Tests(unittest.TestCase):

    # Tests the `one_hot_encode` function, which is passed in as an argument
    def test_one_hot(self, one_hot_function):
        """
        Tests the `one_hot_encode` function, which is passed in as an argument.
        """
        # Test that the generated one-hot labels match the expected one-hot label
        # For all three cases (red, yellow, green)
        try:
            self.assertEqual([1, 0, 0], one_hot_function('red'))
            self.assertEqual([0, 1, 0], one_hot_function('yellow'))
            self.assertEqual([0, 0, 1], one_hot_function('green'))

        # If the function does *not* pass all 3 tests above, it enters this exception
        except self.failureException as e:
            # Print out an error message
            print_fail()
            print("Your function did not return the expected one-hot label.")
            print('\n' + str(e))
            return

        # Print out a "tests passed" message
        print_pass("The `one_hot_encode` function works as expected!")


    def test_red_as_green(self, misclassified_images):
        """
        Tests if any misclassified images are red but mistakenly classified as green
        """
        # Loop through each misclassified image and the labels
        for im, predicted_label, true_label in misclassified_images:

            # Check if the image is one of a red light
            if (true_label == [1, 0, 0]):

                try:
                    # Check that it is NOT labeled as a green light
                    self.assertNotEqual(predicted_label, [0, 0, 1])
                except self.failureException as e:
                    # Print out an error message
                    print_fail()
                    print("Warning: A red light is classified as green.")
                    print('\n' + str(e))
                    return

        # No red lights are classified as green; tests passed
        print_pass("No misclassified Red images are classified as Green!")
