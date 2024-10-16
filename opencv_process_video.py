import cv2
import argparse
import sys
import numpy as np
from student_card import process_student_card, recognize_student_number
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
    output_width = frame_width + frame_height
    output_height = frame_height

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (output_width, output_height))

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Gaussian blur to reduce noise
        # if between(cap, 1000, 3000):
        #     frame = apply_gaussian_blur(frame)
        #
        # # Sharpen the image
        # elif between(cap, 3000, 5000):
        #     frame = apply_sharpening(apply_gaussian_blur(frame))
        #
        # # Sobel edge operator
        # elif between(cap, 5000, 7000):
        #     frame = apply_sobel(frame)
        #
        # # Canny edge operator
        # elif between(cap, 7000, 9000):
        #     frame = apply_canny(frame)
        #
        # # Show DFT spectrum
        # elif between(cap, 9000, 11000):
        #     frame = show_dft_spectrum(frame)
        #
        # # Apply low pass filter
        # elif between(cap, 11000, 13000):
        #     frame = apply_low_pass_filter(frame)
        #
        # # Apply high pass filter
        # elif between(cap, 13000, 15000):
        #     frame = apply_high_pass_filter(frame)
        #
        # # Apply band pass filter
        # elif between(cap, 15000, 17000):
        #     frame = apply_band_pass_filter(frame)

        if between(cap, 0, 5000):  # Between 20s and 25s
            frame = process_student_card(frame, 'templates/student_card.png', 'templates/student_photo.png')

        if between(cap, 5000, 10000):
            frame = recognize_student_number(frame)

        if frame.shape[1] != output_width:
            padding = np.zeros((frame_height, output_width - frame_width, 3), dtype=np.uint8)
            frame = np.hstack((frame, padding))

        out.write(frame)

        frame_count += 1
        if frame_count % 30 == 0:  # Update progress every 30 frames
            progress = (frame_count / total_frames) * 100
            print(f"Processing progress: {progress:.2f}%")

    cap.release()
    out.release()
    print("Video processing completed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenCV video processing')
    parser.add_argument('-i', "--input", help='full path to input video that will be processed')
    parser.add_argument('-o', "--output", help='full path for saving processed video output')
    args = parser.parse_args()

    if args.input is None or args.output is None:
        sys.exit("Please provide path to input and output video files! See --help")

    main(args.input, args.output)