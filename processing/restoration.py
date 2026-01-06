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
    if len(image.shape) == 3:  # RGB
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        restored = wiener(gray, (ksize, ksize))
        return np.uint8(restored)
    else:
        restored = wiener(image, (ksize, ksize))
        return np.uint8(restored)
