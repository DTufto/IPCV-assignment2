import cv2
import numpy as np
from student_card import rotation_invariant_template_matching


def visualize_matching(frame, template, match):
    # Create a copy of the frame and template for visualization
    vis_frame = frame.copy()
    vis_template = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)

    if match:
        x, y, scale, angle, score = match
        h, w = template.shape[:2]

        # Draw bounding box on the frame
        box = np.int32(cv2.boxPoints(((x + w * scale / 2, y + h * scale / 2), (w * scale, h * scale), angle)))
        cv2.drawContours(vis_frame, [box], 0, (0, 255, 0), 2)

        # Draw matching points
        for i in range(4):
            pt_frame = tuple(box[i])
            pt_template = (int(i % 2 * w), int(i // 2 * h))
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            cv2.circle(vis_frame, pt_frame, 5, color, -1)
            cv2.circle(vis_template, pt_template, 5, color, -1)
            cv2.line(vis_frame, pt_frame, (x + pt_template[0], y + pt_template[1]), color, 1)

        # Add score text
        cv2.putText(vis_frame, f"Score: {score:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Resize template to match frame height
    h_frame, w_frame = vis_frame.shape[:2]
    h_temp, w_temp = vis_template.shape[:2]
    vis_template = cv2.resize(vis_template, (int(w_temp * h_frame / h_temp), h_frame))

    # Combine frame and template side by side
    vis = np.hstack((vis_frame, vis_template))
    return vis


def visualize_video_processing(input_video_file: str, template_path: str):
    cap = cv2.VideoCapture(input_video_file)

    # Load the template
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise ValueError(f"Could not read template from {template_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform template matching
        match = rotation_invariant_template_matching(frame, template)

        # Create visualization
        vis = visualize_matching(frame, template, match)

        # Display the visualization
        cv2.imshow('Template Matching Visualization', vis)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    visualize_video_processing('videos/student_card_scaled.mp4', 'templates/student_photo.png')