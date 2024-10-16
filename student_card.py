import cv2
import numpy as np
import pytesseract
from pytesseract import Output


def recognize_student_number(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    data = pytesseract.image_to_data(threshold, output_type=Output.DICT)
    student_number = ''

    for i, text in enumerate(data['text']):
        if len(text) == 7 and text.isdigit():
            student_number = text
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if student_number:
        cv2.putText(frame, f"Student Number: {student_number}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

def detect_and_match_orb(image, template, min_matches=10):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(template, None)
    kp2, des2 = orb.detectAndCompute(image, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) > min_matches:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = template.shape[:2]

        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        return dst, True
    else:
        return None, False


def detect_photo_in_card(card_image, photo_template):
    card_gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
    photo_gray = cv2.cvtColor(photo_template, cv2.COLOR_BGR2GRAY)
    scale_factors = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

    best_match = None
    best_scale = None
    best_val = -1

    for scale in scale_factors:
        resized_template = cv2.resize(photo_gray, None, fx=scale, fy=scale)
        if resized_template.shape[0] > card_gray.shape[0] or resized_template.shape[1] > card_gray.shape[1]:
            continue

        result = cv2.matchTemplate(card_gray, resized_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if max_val > best_val:
            best_val = max_val
            best_match = max_loc
            best_scale = scale

    if best_val > 0.5:
        h, w = photo_template.shape[:2]
        return best_match, (int(h * best_scale), int(w * best_scale))
    else:
        return None, None


def process_student_card(frame, card_template_path, photo_template_path):
    card_template = cv2.imread(card_template_path, cv2.IMREAD_GRAYSCALE)
    photo_template = cv2.imread(photo_template_path)

    if card_template is None or photo_template is None:
        raise ValueError(f"Could not read templates")

    if frame is None or frame.size == 0:
        raise ValueError("Input frame is empty")

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    card_template_name = card_template_path.split('/')[-1].split('.')[0]

    card_corners, detected = detect_and_match_orb(frame_gray, card_template)

    if not detected:
        cv2.putText(frame, f"{card_template_name} Not Detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return frame

    cv2.polylines(frame, [np.int32(card_corners)], True, (0, 255, 0), 3, cv2.LINE_AA)

    x, y, w, h = cv2.boundingRect(np.int32(card_corners))
    card_region = frame[y:y + h, x:x + w]

    if card_region.size == 0:
        cv2.putText(frame, "Invalid card region", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return frame

    photo_loc, photo_size = detect_photo_in_card(card_region, photo_template)

    if photo_loc is not None:
        ph_x, ph_y = photo_loc
        ph_h, ph_w = photo_size
        cv2.rectangle(card_region, (ph_x, ph_y), (ph_x + ph_w, ph_y + ph_h), (255, 0, 0), 2)
        cv2.putText(frame, "Photo Detected", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    else:
        cv2.putText(frame, "Photo Not Detected", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.putText(frame, f"{card_template_name} Detected", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame