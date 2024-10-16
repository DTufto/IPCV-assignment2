import cv2
import numpy as np


def apply_sobel(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel = np.uint8(255 * sobel / np.max(sobel))
    return cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)

def apply_canny(frame, low_threshold=100, high_threshold=200):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)