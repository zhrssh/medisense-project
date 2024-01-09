import cv2
import numpy as np

from skimage import img_as_ubyte
from skimage.morphology import skeletonize


def predict(path):
    """
    Performs preprocessing to the image.

    Params:
        path: path to image

    Returns:
        img: preprocessed image
    """
    # Reads image
    img = cv2.imread(path)

    # Converts to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoises the image
    denoised = cv2.bilateralFilter(gray, 15, 80, 80)

    # Adjusts the constrast of the image
    adjusted = cv2.addWeighted(
        denoised, 1.3, np.zeros(denoised.shape, denoised.dtype), 0, -50
    )

    # Create truth masks
    kernel = np.ones((3, 3), np.uint8)
    hand_mask = _create_hand_mask(adjusted.copy(), kernel)
    truth_mask = _create_vein_mask(adjusted.copy(), hand_mask, kernel)

    # Skeletonize image
    skeleton = skeletonize(truth_mask)

    # Converts into cv image
    skeleton = img_as_ubyte(skeleton)

    return skeleton


def _create_hand_mask(image, kernel=(3, 3)):
    """
    Creates an eroded mask of the hand. This ignors the edges of the hands
    and keeps only what's inside of the hand.

    Params:
      image: Image to create the hand mask from
      kernel: Used for morphological operations

    Returns:
      hand_mask: Mask of the hand
    """
    # Find contours and get only the largest contour
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest_contour = max(contours, key=cv2.contourArea)

    # Create hand mask
    hand_mask = np.zeros_like(image)
    cv2.fillPoly(hand_mask, [largest_contour], 255)
    hand_mask = cv2.erode(hand_mask, kernel, iterations=15)

    return hand_mask


def _create_vein_mask(image, hand_mask, kernel=(3, 3)):
    """
    Creates the vein mask used for training the segmentation model.

    Params:
      image: Image to create vein mask from
      hand_mask: Hand mask for isolating the veins
      kernel: Used for morphological operations

    Returns:
      vein_mask: Mask of the veins
    """

    # Applies adaptive thresholding and normalize the output 27, 5
    thresholded = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 27, 5
    )

    # # Apply dilation to connect edges
    dilated = cv2.dilate(thresholded, kernel, iterations=2)

    # Apply erosion to remove small objects
    eroded = cv2.erode(dilated, kernel, iterations=1)

    # Find contours of the eroded image
    contours, _ = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Create vein mask
    vein_mask = np.zeros_like(image)
    cv2.fillPoly(vein_mask, contours, 255)

    # Isolates the veins using the hand_mask
    vein_mask = cv2.bitwise_and(vein_mask, hand_mask)

    # Find contours of the final mask image
    contours, _ = cv2.findContours(vein_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    large_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 200:
            large_contours.append(contour)

    # Creates the final ground truth mask
    final_mask = np.zeros_like(image)
    cv2.fillPoly(final_mask, large_contours, 255)

    # Smoothen out final mask
    final_mask = cv2.morphologyEx(
        final_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )

    return final_mask
