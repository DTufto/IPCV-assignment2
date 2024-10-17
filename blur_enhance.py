import cv2
import numpy as np


def apply_gaussian_blur(frame, kernel_size=(5, 5), sigma=3):
    return cv2.GaussianBlur(frame, kernel_size, sigma)


def apply_sharpening(frame, kernel_strength=1):
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]]) * kernel_strength
    return cv2.filter2D(frame, -1, kernel)


def enhance_image(frame, blur_kernel_size=(5, 5), blur_sigma=0, sharpen_strength=1):
    blurred = apply_gaussian_blur(frame, blur_kernel_size, blur_sigma)
    sharpened = apply_sharpening(blurred, sharpen_strength)
    return sharpened
