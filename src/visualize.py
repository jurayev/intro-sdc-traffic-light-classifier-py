import cv2
import matplotlib.pyplot as plt


def get_sorted_images_by_class(images):
    r_images = []
    y_images = []
    g_images = []

    for img in images:
        if img[1] in ['red', [1, 0, 0]]:
            r_images.append(img)
        elif img[1] in ['green', [0, 0, 1]]:
            g_images.append(img)
        else:
            y_images.append(img)
    return r_images, y_images, g_images


def show_images(image_one, image_two, image_three):
    """
    Plots three images with label and shape attributes
    """
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
    ax1.set_title("%s with shape %s" % (image_one[1], image_one[0].shape))
    ax1.imshow(image_one[0])
    ax2.set_title("%s with shape %s" % (image_two[1], image_two[0].shape))
    ax2.imshow(image_two[0])
    ax3.set_title("%s with shape %s" % (image_three[1], image_three[0].shape))
    ax3.imshow(image_three[0])


def show_hsv_image(image):
    """
    Converts and image to HSV colorspace. Visualize the individual color channels
    """
    test_im = image[0]
    test_label = image[1]

    # Convert to HSV
    hsv = cv2.cvtColor(test_im, cv2.COLOR_RGB2HSV)

    # Print image label
    print('Label [red, yellow, green]: ' + str(test_label))

    # HSV channels
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # Plot the original image and the three channels
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 10))
    ax1.set_title('Standardized image')
    ax1.imshow(test_im)
    ax2.set_title('H channel')
    ax2.imshow(h, cmap='gray')
    ax3.set_title('S channel')
    ax3.imshow(s, cmap='gray')
    ax4.set_title('V channel')
    ax4.imshow(v, cmap='gray')
