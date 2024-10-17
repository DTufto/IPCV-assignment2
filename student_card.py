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
    scale_factors = np.arange(0.5, 4.6, 0.1)

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

    h, w = photo_template.shape[:2]
    if best_scale:
        return best_match, (int(h * best_scale), int(w * best_scale))
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


def apply_optical_flow(prev_frame, curr_frame):
    if prev_frame.shape != curr_frame.shape:
        curr_frame = cv2.resize(curr_frame, (prev_frame.shape[1], prev_frame.shape[0]))

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    mask = np.zeros_like(curr_frame)
    mask[..., 2] = 255
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    rgb = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)
    result = cv2.add(rgb, mask)

    step = 16
    for y in range(0, curr_frame.shape[0], step):
        for x in range(0, curr_frame.shape[1], step):
            fx, fy = flow[y, x]
            cv2.arrowedLine(result, (x, y), (int(x + fx), int(y + fy)), (0, 255, 0), 1, tipLength=0.5)

    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
