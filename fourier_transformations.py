import cv2
import numpy as np


def apply_dft(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    return dft_shift, magnitude_spectrum


def create_magnitude_spectrum_image(magnitude_spectrum):
    magnitude_spectrum_normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.applyColorMap(magnitude_spectrum_normalized, cv2.COLORMAP_PARULA)


def show_dft_spectrum(frame):
    _, magnitude_spectrum = apply_dft(frame)
    magnitude_spectrum_image = create_magnitude_spectrum_image(magnitude_spectrum)
    height, width = frame.shape[:2]
    magnitude_spectrum_image_resized = cv2.resize(magnitude_spectrum_image, (height, height))
    title = "Magnitude Spectrum"
    frame = cv2.resize(frame, (width - height, height))
    cv2.putText(magnitude_spectrum_image_resized, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    combined = np.hstack((frame, magnitude_spectrum_image_resized))

    return combined


def apply_filter(dft_shift, filter_mask):
    fshift = dft_shift * filter_mask[:, :, np.newaxis]
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def create_low_pass_filter(shape, radius):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.float32)
    cv2.circle(mask, (ccol, crow), radius, 1, -1)
    return mask


def create_high_pass_filter(shape, radius):
    return 1 - create_low_pass_filter(shape, radius)


def create_band_pass_filter(shape, inner_radius, outer_radius):
    low_pass = create_low_pass_filter(shape, outer_radius)
    high_pass = create_high_pass_filter(shape, inner_radius)
    return low_pass * high_pass


def apply_low_pass_filter(frame, radius=30):
    dft_shift, _ = apply_dft(frame)
    rows, cols = frame.shape[:2]
    mask = create_low_pass_filter((rows, cols), radius)
    result = apply_filter(dft_shift, mask)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


def apply_high_pass_filter(frame, radius=30):
    dft_shift, _ = apply_dft(frame)
    rows, cols = frame.shape[:2]
    mask = create_high_pass_filter((rows, cols), radius)
    result = apply_filter(dft_shift, mask)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


def apply_band_pass_filter(frame, inner_radius=10, outer_radius=50):
    dft_shift, _ = apply_dft(frame)
    rows, cols = frame.shape[:2]
    mask = create_band_pass_filter((rows, cols), inner_radius, outer_radius)
    result = apply_filter(dft_shift, mask)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
