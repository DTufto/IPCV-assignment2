import cv2
import numpy as np
import pytesseract
from pytesseract import Output


def recognize_text(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return pytesseract.image_to_data(threshold, output_type=Output.DICT)


def rotation_size_invariant_template_matching(image, template, scale_range=(0.5, 1.5), scale_steps=20,
                                              rotation_range=(-180, 180), rotation_steps=36, threshold=0.7):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    template_height, template_width = gray_template.shape

    best_matches = []

    scales = np.linspace(scale_range[0], scale_range[1], scale_steps)
    angles = np.linspace(rotation_range[0], rotation_range[1], rotation_steps)

    for scale in scales:
        scaled_template = cv2.resize(gray_template, None, fx=scale, fy=scale)

        for angle in angles:
            rotation_matrix = cv2.getRotationMatrix2D((scaled_template.shape[1] / 2, scaled_template.shape[0] / 2),
                                                      angle, 1)
            rotated_template = cv2.warpAffine(scaled_template, rotation_matrix,
                                              (scaled_template.shape[1], scaled_template.shape[0]))

            result = cv2.matchTemplate(gray_image, rotated_template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= threshold)

            for pt in zip(*locations[::-1]):
                score = result[pt[1], pt[0]]
                best_matches.append((pt[0], pt[1], scale, angle, score))

    best_matches.sort(key=lambda x: x[4], reverse=True)
    final_matches = []

    while best_matches:
        current_match = best_matches.pop(0)
        final_matches.append(current_match)

        best_matches = [
            match for match in best_matches
            if abs(match[0] - current_match[0]) > template_width // 2 or
               abs(match[1] - current_match[1]) > template_height // 2
        ]

    return final_matches


def recognize_logo(frame, template_path='templates/UT_Logo_Black_EN.png'):
    template = cv2.imread(template_path)
    matches = rotation_size_invariant_template_matching(frame, template, scale_range=(0.1, 1.0), threshold=0.6)

    if matches:
        best_match = matches[0]
        x, y, scale, angle, score = best_match
        h, w = template.shape[:2]
        box = np.int32(cv2.boxPoints(((x + w * scale / 2, y + h * scale / 2), (w * scale, h * scale), angle)))
        return box
    return None


def process_student_card(frame):
    data = recognize_text(frame)
    student_number = ""

    for i, text in enumerate(data['text']):
        if len(text) == 7 and text[1:].isdigit():
            student_number = text
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if student_number:
        cv2.putText(frame, f"Student Number: {student_number}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    logo_box = recognize_logo(frame)
    if logo_box is not None:
        cv2.drawContours(frame, [logo_box], 0, (255, 0, 0), 2)
        cv2.putText(frame, "Logo Detected", (logo_box[0][0], logo_box[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return frame