import cv2
import numpy as np
from sklearn.cluster import DBSCAN


def detect_objects(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0, 50, 50])
    upper_bound = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 100
    objects = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fast = cv2.FastFeatureDetector_create()
    keypoints = fast.detect(gray, None)

    for cnt in objects:
        x, y, w, h = cv2.boundingRect(cnt)
        cnt_keypoints = [kp for kp in keypoints if x <= kp.pt[0] < x + w and y <= kp.pt[1] < y + h]
        cnt_points = np.float32([kp.pt for kp in cnt_keypoints]).reshape(-1, 2)
        if len(cnt_points) > 10:
            dbscan = DBSCAN(eps=30, min_samples=5)
            dbscan.fit(cnt_points)

    return objects, keypoints


def match_objects(objects1, keypoints1, objects2, keypoints2, frame1, frame2):
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matched_objects = []

    for i, obj1 in enumerate(objects1):
        x1, y1, w1, h1 = cv2.boundingRect(obj1)
        roi1 = frame1[y1:y1 + h1, x1:x1 + w1]

        hist1 = cv2.calcHist([roi1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)

        kp1 = [kp for kp in keypoints1 if x1 <= kp.pt[0] < x1 + w1 and y1 <= kp.pt[1] < y1 + h1]
        _, des1 = orb.compute(roi1, kp1)

        best_match = None
        max_score = -1
        for j, obj2 in enumerate(objects2):
            x2, y2, w2, h2 = cv2.boundingRect(obj2)
            roi2 = frame2[y2:y2 + h2, x2:x2 + w2]

            hist2 = cv2.calcHist([roi2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
            hist_correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

            kp2 = [kp for kp in keypoints2 if x2 <= kp.pt[0] < x2 + w2 and y2 <= kp.pt[1] < y2 + h2]
            _, des2 = orb.compute(roi2, kp2)

            if des1 is not None and des2 is not None:
                matches = bf.match(des1, des2)
                feature_score = len(matches) / max(len(kp1), len(kp2))
            else:
                feature_score = 0

            combined_score = 0.2 * hist_correlation + 0.8 * feature_score

            if combined_score > max_score:
                max_score = combined_score
                best_match = j

        if best_match is not None:
            matched_objects.append((i, best_match))

    return matched_objects


def draw_objects(frame1, frame2, objects1, objects2, matched_objects, colors):
    for (i, j), color in zip(matched_objects, colors):
        cv2.drawContours(frame1, [objects1[i]], 0, color, 2)
        cv2.drawContours(frame2, [objects2[j]], 0, color, 2)

        M1 = cv2.moments(objects1[i])
        M2 = cv2.moments(objects2[j])

        if M1['m00'] != 0 and M2['m00'] != 0:
            cx1 = int(M1['m10'] / M1['m00'])
            cy1 = int(M1['m01'] / M1['m00'])
            cx2 = int(M2['m10'] / M2['m00'])
            cy2 = int(M2['m01'] / M2['m00'])

            cv2.line(frame1, (cx1, cy1), (frame1.shape[1], cy1), color, 2)
            cv2.line(frame2, (0, cy2), (cx2, cy2), color, 2)

    return frame1, frame2


def process_videos(current_video_path, second_video_path, start_time, duration, target_size=(1280, 720)):
    cap1 = cv2.VideoCapture(current_video_path)
    cap2 = cv2.VideoCapture(second_video_path)

    if not cap1.isOpened() or not cap2.isOpened():
        print(f"Error: Unable to open video files.")
        return None

    fps = int(round(cap1.get(cv2.CAP_PROP_FPS)))
    start_frame = int(start_time * fps)
    end_frame = start_frame + int(duration * fps)

    cap1.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)

    target_width, target_height = target_size
    combined_frames = []

    for _ in range(start_frame, end_frame):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        frame1 = cv2.resize(frame1, (target_width // 2, target_height), interpolation=cv2.INTER_CUBIC)
        frame2 = cv2.resize(frame2, (target_width // 2, target_height), interpolation=cv2.INTER_CUBIC)

        try:
            objects1, keypoints1 = detect_objects(frame1)
            objects2, keypoints2 = detect_objects(frame2)

            matched_objects = match_objects(objects1, keypoints1, objects2, keypoints2, frame1, frame2)

            colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in
                      range(len(matched_objects))]

            frame1, frame2 = draw_objects(frame1, frame2, objects1, objects2, matched_objects, colors)

            combined_frame = np.hstack((frame1, frame2))
            combined_frames.append(combined_frame)
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            combined_frame = np.hstack((frame1, frame2))
            combined_frames.append(combined_frame)

    cap1.release()
    cap2.release()
    return combined_frames