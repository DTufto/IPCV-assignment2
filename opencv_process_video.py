import cv2
import argparse
import sys

from object_matching import process_videos
from student_card import process_student_card, recognize_student_number, apply_optical_flow
from edge_detection import apply_canny, apply_sobel
from blur_enhance import apply_gaussian_blur, apply_sharpening
from fourier_transformations import (
    show_dft_spectrum, apply_low_pass_filter,
    apply_high_pass_filter, apply_band_pass_filter
)


def between(cap, lower: int, upper: int) -> bool:
    return lower <= int(cap.get(cv2.CAP_PROP_POS_MSEC)) < upper


def main(input_video_file: str, output_video_file: str) -> None:
    cap = cv2.VideoCapture(input_video_file)
    fps = int(round(cap.get(5)))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    upscale_factor = 3
    new_width = int(frame_width * upscale_factor)
    new_height = int(frame_height * upscale_factor)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (new_width, new_height))

    prev_frame = None

    while cap.isOpened() and cap.get(cv2.CAP_PROP_POS_MSEC) < 40000:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        if between(cap, 0, 1000):
            cv2.putText(frame, "Original Video", (10, new_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Gaussian blur to reduce noise
        elif between(cap, 1000, 3000):
            kernel_size = (5, 5)
            sigma = 3
            frame = apply_gaussian_blur(frame, kernel_size, sigma)
            cv2.putText(frame, f"Gaussian Blur (kernel={kernel_size}, sigma={sigma})", (10, new_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Sharpen the image
        elif between(cap, 3000, 5000):
            kernel_strength = 1
            frame = apply_sharpening(apply_gaussian_blur(frame), kernel_strength)
            cv2.putText(frame, f"Sharpen Image (strength={kernel_strength})", (10, new_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Sobel edge operator
        elif between(cap, 5000, 7500):
            frame = apply_sobel(frame)
            cv2.putText(frame, "Sobel edge (ksize=3)", (10, new_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Canny edge operator
        elif between(cap, 7500, 10000):
            low_threshold = 100
            high_threshold = 200
            frame = apply_canny(frame, low_threshold, high_threshold)
            cv2.putText(frame, f"Canny edge (low={low_threshold}, high={high_threshold})", (10, new_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Show DFT spectrum
        elif between(cap, 10000, 12500):
            frame = show_dft_spectrum(frame)
            cv2.putText(frame, "DFT spectrum", (10, new_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Apply low pass filter
        elif between(cap, 12500, 15000):
            radius = 30
            frame = apply_low_pass_filter(frame, radius)
            cv2.putText(frame, f"Low-pass filter (radius={radius})", (10, new_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Apply high pass filter
        elif between(cap, 15000, 17500):
            radius = 30
            frame = apply_high_pass_filter(frame, radius)
            cv2.putText(frame, f"High-pass filter (radius=1 - low pass filter)", (10, new_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Apply band pass filter
        elif between(cap, 17500, 20000):
            inner_radius = 10
            outer_radius = 50
            frame = apply_band_pass_filter(frame, inner_radius, outer_radius)
            cv2.putText(frame, f"Band-pass filter (inner={inner_radius}, outer={outer_radius})", (10, new_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Recognize UT logo
        if between(cap, 20000, 25000):
            frame = process_student_card(frame, 'templates/student_card.png', 'templates/UT_logo.png')
            cv2.putText(frame, "Template match UT-logo", (10, new_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Recognize student number
        if between(cap, 25000, 30000):
            frame = recognize_student_number(frame)
            cv2.putText(frame, "Number recognition with pytesseract", (10, new_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Recognize photo
        if between(cap, 30000, 35000):
            frame = process_student_card(frame, 'templates/student_card.png', 'templates/student_photo.png')
            cv2.putText(frame, "Template match photo", (10, new_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Apply optical flow
        if between(cap, 35000, 40000):
            if prev_frame is not None:
                frame = apply_optical_flow(prev_frame, frame)
                cv2.putText(frame, "Optical flow (Farneback method)", (10, new_height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            prev_frame = frame.copy()

        out.write(frame)

    start_time = 40
    duration = 20
    combined_frames = process_videos(input_video_file, 'templates/freestyle.mp4', start_time, duration,
                                     target_size=(new_width, new_height))

    if combined_frames:
        for frame in combined_frames:
            cv2.putText(frame, "Object/feature matching between two videos using color histograms, FAST, and ORB. Matched objects are connected with lines.", (10, new_height - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            out.write(frame)

    cap.release()
    out.release()
    print("Video processing completed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenCV video processing')
    parser.add_argument('-i', "--input", help='full path to input video that will be processed')
    parser.add_argument('-o', "--output", help='full path for saving processed video output')
    args = parser.parse_args()

    if args.input is None or args.output is None:
        sys.exit("Please provide paths to input and output! See --help")

    main(args.input, args.output)
