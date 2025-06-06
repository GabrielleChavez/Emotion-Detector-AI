import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace


# ----------------------------
# Helpers for Eye‐Contact & Smile
# ----------------------------

def normalized_iris_ratio(iris_idx, left_corner_idx, right_corner_idx, landmarks, ih, iw):
    """
    Normalize the iris distance from eye‐center to [-1 .. 1]
    Parameters:
        iris_idx: index of the iris landmark
        left_corner_idx: index of the left corner landmark
        right_corner_idx: index of the right corner landmark
        landmarks: list of landmarks
        ih: height of the image
        iw: width of the image
    Returns:
        Normalized iris distance from eye‐center to [-1 .. 1]
    """
    # Get pixel coords
    ix = int(landmarks[iris_idx].x * iw)
    iy = int(landmarks[iris_idx].y * ih)
    lx = int(landmarks[left_corner_idx].x * iw)
    rx = int(landmarks[right_corner_idx].x * iw)

    # Eye‐center x = (lx + rx)/2
    eye_center_x = (lx + rx) / 2.0

    # Normalize iris distance from eye‐center to [-1 .. 1]
    eye_width = abs(rx - lx)
    if eye_width < 1:
        return 0.0
    return (ix - eye_center_x) / (eye_width / 2.0)

def eye_contact(landmarks, image_shape):
    """
    Estimate eye contact by checking iris position relative to eye bounding box.
    Returns True if both eyes approximately look straight at camera.
    Parameters:
        landmarks: list of landmarks
        image_shape: shape of the image
    Returns:
        True if both eyes approximately look straight at camera, False otherwise
    """
    ih, iw = image_shape[:2]

    # Choose iris landmarks
    left_ratio = normalized_iris_ratio(469, 33, 133, landmarks, ih, iw)
    right_ratio = normalized_iris_ratio(474, 362, 263, landmarks, ih, iw)

    # Measures if face is looking forward
    return (abs(left_ratio) < 0.65) and (abs(right_ratio) < 0.65)


def mouth_aspect_ratio(landmarks, image_shape):
    """
    Compute a simple Mouth Aspect Ratio (MAR). If MAR > threshold, smile = True.
    Using landmarks:
      - Upper lip center ~ index 13
      - Lower lip center ~ index 14
      - Mouth left corner ~ index 61
      - Mouth right corner ~ index 291
    Parameters:
        landmarks: list of landmarks
        image_shape: shape of the image
    Returns:
        MAR value 
    """
    ih, iw = image_shape[:2]
    pt = lambda idx: np.array([landmarks[idx].x * iw, landmarks[idx].y * ih])

    upper = pt(13)
    lower = pt(14)
    left = pt(61)
    right = pt(291)

    vertical_dist = np.linalg.norm(upper - lower)
    horizontal_dist = np.linalg.norm(left - right)
    if horizontal_dist < 1:
        return 0.0
    return vertical_dist / horizontal_dist


# ----------------------------
# Initialize MediaPipe Face Mesh
# ----------------------------
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    refine_landmarks=True,      # we get iris landmarks from 'refine'
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ----------------------------
# Start Video Capture
# ----------------------------
cap = cv2.VideoCapture(0)  # change index if multiple cameras

if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

# ----------------------------
# Main Loop
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip horizontally for natural (mirror) view
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform face + landmark detection
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            ih, iw = frame.shape[:2]

            # Compute face bounding box from landmarks
            xs = [lm.x for lm in face_landmarks.landmark]
            ys = [lm.y for lm in face_landmarks.landmark]
            x_min = int(min(xs) * iw) - 10
            x_max = int(max(xs) * iw) + 10
            y_min = int(min(ys) * ih) - 10
            y_max = int(max(ys) * ih) + 10

            # Clip to image boundaries
            x_min = max(x_min, 0)
            y_min = max(y_min, 0)
            x_max = min(x_max, iw - 1)
            y_max = min(y_max, ih - 1)

            # Crop face for emotion recognition
            face_roi = frame[y_min:y_max, x_min:x_max] #np.array((x_max-x_min, y_max-y_min)) #
        
            emotion_label = "Neutral"
            try:
                # DeepFace returns a dict, e.g. {'emotion': {'happy': 0.95, ...}, ...}
                analysis = DeepFace.analyze(face_roi, actions=["emotion"], enforce_detection=False)
                # 'dominant_emotion' already provided
                emotion_label = analysis[0]["dominant_emotion"].capitalize()
            except Exception as e:
                # If DeepFace fails, default to “Neutral”
                print("Error: ", e)
                exit()
                emotion_label = "Neutral"

            # Check eye contact
            is_eye_contact = eye_contact(face_landmarks.landmark, frame.shape)
            eye_contact_text = "Yes" if is_eye_contact else "No"

            # Check smiling via MAR
            mar = mouth_aspect_ratio(face_landmarks.landmark, frame.shape)
            is_smiling = mar > 0.10
            smiling_text = "Yes" if is_smiling else "No"

            # Draw bounding box and state texts
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 200, 0), 2)

            # Prepare overlay text
            text_1 = f"Emotion: {emotion_label}"
            text_2 = f"Eye Contact: {eye_contact_text}"
            text_3 = f"Smiling: {smiling_text}"

            # Put text above the face box
            cv2.putText(frame, text_1, (x_min, y_min - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, text_2, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, text_3, (x_min, y_min + (y_max - y_min) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


    # Show the resulting frame
    cv2.imshow("Live Emotion/Eye Contact/Smile Detector", frame)

    # Press 'q' to break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
