import cv2
import numpy as np
from scipy.signal import wiener

# Mean Filtering
def mean_filter(image, ksize=3):
    return cv2.blur(image, (ksize, ksize))

# Median Filtering
def median_filter(image, ksize=3):
    return cv2.medianBlur(image, ksize)

# Wiener / Adaptive Filtering
def wiener_filter(image, ksize=3):
    img = image.astype(np.float32)

    filtered = wiener(img, (ksize, ksize))

    filtered = np.clip(filtered, 0, 255)

    return filtered.astype(np.uint8)