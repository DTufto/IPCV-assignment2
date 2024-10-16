import cv2
import numpy as np


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


def process_student_card(frame, template_path):
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise ValueError(f"Could not read template from {template_path}")

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    template_name = template_path.split('/')[-1].split('.')[0]

    card_corners, detected = detect_and_match_orb(frame_gray, template)

    if not detected:
        cv2.putText(frame, f"{template_name} Not Detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return frame

    cv2.polylines(frame, [np.int32(card_corners)], True, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.putText(frame, f"{template_name} Detected", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame