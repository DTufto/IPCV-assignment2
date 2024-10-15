import cv2
import pytesseract
from pytesseract import Output


def recognize_text(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return pytesseract.image_to_data(threshold, output_type=Output.DICT)


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

    return frame
