import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Open the webcam
cap = cv2.VideoCapture(0)

# with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

def blur_image(image, points: list):
    h_img, w_img = image.shape[:2]  # Get image dimensions
    x, y, w, h = points

    # Ensure x, y, w, h are within bounds
    x = max(0, min(x, w_img - 1))
    y = max(0, min(y, h_img - 1))
    w = max(0, min(w, w_img - x))  # Ensure width doesn't exceed image
    h = max(0, min(h, h_img - y))  # Ensure height doesn't exceed image

    try:
        roi = image[y:y+h, x:x+w]

        # Apply Gaussian Blur
        blurred_roi = cv2.blur(roi, (200, 200), 0)

        # Replace the original region with blurred region
        image[y:y+h, x:x+w] = blurred_roi
        
        return image
    except Exception as e:
        print("Error occurred:", e)
        return cv2.blur(image, (200, 200), 0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect faces
    results = face_detection.process(rgb_frame)

    # Draw bounding boxes if faces are detected
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, width, height = (int(bboxC.xmin * w), int(bboxC.ymin * h),
                                    int(bboxC.width * w), int(bboxC.height * h))

            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 3)
            prev_points = [x, y, width, height]
            # blur_image(frame, [x, y, width,  height])

    # Display the output
    cv2.imshow('MediaPipe Face Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()