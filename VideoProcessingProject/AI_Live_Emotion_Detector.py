import cv2
import numpy as np
import mediapipe as mp
from google import genai
import tkinter as tk
from tkinter import scrolledtext, Entry, Button, END, Text
import threading
import queue
import time
import os # For environment variables
from deepface import DeepFace

################################################
#                 FUNCTIONS                    #
################################################

try:
    # Ensure the API key is set before configuring
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    client = genai.Client(api_key="AIzaSyA1bSl6VT0YXC9tujsY-E6gkbJW3sippV0")
    print("✅ Gemini API configured successfully.")
except Exception as e:
    print(f"❌ Failed to configure Gemini API: {e}")
    print("Please ensure you have set the GOOGLE_API_KEY environment variable and it is valid.")
    # You might want to exit or disable AI functionality if the API fails
    exit()


# ------------------------ AI INTERACTION LOGIC ------------------------
def generate_ai_response(emotion, eye_contact, smiling, user_question=None):
    """
    Generates a response from the Gemini model, incorporating facial data. It gathers user's current emotion, eye contact, and smiling status, 
    then generates a response based on the user's question or proactively observes the user's facial expressions.
    Parameters:
        emotion: str, current emotion of the user
        eye_contact: bool, whether the user is making eye contact
        smiling: bool, whether the user is smiling
        user_question: str, user's question (optional)
    Returns:
        str, AI response
    """
    
    # Construct the prompt with facial context
    prompt_parts = []
    if user_question:
        prompt_parts.append(f"The user's current emotion is {emotion}, they are {'making' if eye_contact else 'not making'} eye contact, and they are {'smiling' if smiling else 'not smiling'}.")
        prompt_parts.append(f"The user asks: '{user_question}'")
        prompt_parts.append("Given their facial expression and their question, respond in a helpful, empathetic, and concise manner. Keep your response brief.")
    else:
        # Proactive observation if no specific question
        if emotion == "Happy":
            prompt_parts.append("The user looks very happy. Respond with a positive and engaging observation.")
        elif emotion == "Sad":
            prompt_parts.append("The user seems a bit sad. Offer a comforting or understanding remark.")
        elif emotion == "Neutral":
            prompt_parts.append("The user has a neutral expression. Offer a general, friendly greeting or ask how you can assist.")
        elif emotion == "Angry":
            prompt_parts.append("The user looks angry. Suggest taking a moment to calm down or ask if something is wrong gently.")
        elif emotion == "Surprise":
            prompt_parts.append("The user looks surprised. Ask what surprised them or express curiosity.")
        elif emotion == "Fear":
            prompt_parts.append("The user appears fearful. Offer reassurance or ask if they are okay.")
        elif emotion == "Disgust":
            prompt_parts.append("The user shows disgust. Acknowledge their expression without being judgmental.")
        else:
             prompt_parts.append("The user's expression is detected as None. Offer a general greeting.")

        # Add eye contact and smiling context to observations
        if eye_contact:
            prompt_parts.append("They are also making eye contact.")
        if smiling:
            prompt_parts.append("They are also smiling.")

    try:
        # Use generate_content for text-only input to Gemini
        # It's good practice to add generation_config for more controlled output
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt_parts,
        )
        return response.text
    except Exception as e:
        print(f"Gemini API call failed: {e}")
        # Check for specific errors, e.g., rate limits, invalid API key
        return "I'm having trouble connecting right now. Please try again later."


# ----------------------------  EYE CONTACT AND SMILE DETECTION ----------------------------

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
    ix = int(landmarks[iris_idx].x * iw)
    iy = int(landmarks[iris_idx].y * ih)
    lx = int(landmarks[left_corner_idx].x * iw)
    rx = int(landmarks[right_corner_idx].x * iw)
    eye_center_x = (lx + rx) / 2.0
    eye_width = abs(rx - lx)
    if eye_width < 1: return 0.0
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
    left_ratio = normalized_iris_ratio(469, 33, 133, landmarks, ih, iw)
    right_ratio = normalized_iris_ratio(474, 362, 263, landmarks, ih, iw)
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
    if horizontal_dist < 1: return 0.0
    return vertical_dist / horizontal_dist

################################################
#                 MAIN LOGIC                    #
################################################


# ----------------------------
# Global variables for communication between threads
# ----------------------------
latest_emotion = "Neutral"
latest_eye_contact = False
latest_smiling = False
question_queue = queue.Queue() # To send questions from GUI to AI processing
response_queue = queue.Queue() # To send AI responses back to GUI
stop_video_flag = threading.Event() # To signal video thread to stop

# ----------------------------
# Function for Video Processing (runs in a separate thread)
# ----------------------------
def video_processing_thread():
    global latest_emotion, latest_eye_contact, latest_smiling
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        stop_video_flag.set()
        return

    while not stop_video_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            # Only process the first detected face for simplicity
            face_landmarks = results.multi_face_landmarks[0]
            ih, iw = frame.shape[:2]
            xs = [lm.x for lm in face_landmarks.landmark]
            ys = [lm.y for lm in face_landmarks.landmark]
            x_min = int(min(xs) * iw) - 10
            x_max = int(max(xs) * iw) + 10
            y_min = int(min(ys) * ih) - 10
            y_max = int(max(ys) * ih) + 10

            x_min = max(x_min, 0)
            y_min = max(y_min, 0)
            x_max = min(x_max, iw - 1)
            y_max = min(y_max, ih - 1)

            if x_max - x_min > 0 and y_max - y_min > 0:
                face_roi = frame[y_min:y_max, x_min:x_max]
                current_emotion = "Neutral"
                try:
                    analysis = DeepFace.analyze(face_roi, actions=["emotion"], enforce_detection=False)
                    current_emotion = analysis[0]["dominant_emotion"].capitalize()
                except Exception as e:
                    # print(f"DeepFace analysis error: {e}")
                    current_emotion = "Neutral"

                current_eye_contact = eye_contact(face_landmarks.landmark, frame.shape)
                current_smiling = mouth_aspect_ratio(face_landmarks.landmark, frame.shape) > 0.10

                # Update global states
                latest_emotion = current_emotion
                latest_eye_contact = current_eye_contact
                latest_smiling = current_smiling

                # Draw bounding box and state texts on video frame
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 200, 0), 2)
                cv2.putText(frame, f"Emotion: {latest_emotion}", (x_min, y_min - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Eye Contact: {'Yes' if latest_eye_contact else 'No'}", (x_min, y_min - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Smiling: {'Yes' if latest_smiling else 'No'}", (x_min, y_min + (y_max - y_min) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show the resulting frame (always show, even if no face detected)
        cv2.imshow("Live Emotion/Eye Contact/Smile Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_video_flag.set()
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Video processing stopped.")

# ----------------------------
# Function for AI Processing (runs in a separate thread)
# ----------------------------
def ai_processing_thread():
    last_observation_time = time.time() # To control frequency of proactive observations
    observation_interval = 10 # seconds between proactive observations

    while not stop_video_flag.is_set():
        user_question = None
        try:
            # Get a question from the queue, non-blocking with a short timeout
            user_question = question_queue.get(timeout=0.05) # Very short timeout for responsiveness
        except queue.Empty:
            pass # No user question

        # Generate AI response either for a question or as a proactive observation
        if user_question:
            response = generate_ai_response(latest_emotion, latest_eye_contact, latest_smiling, user_question)
            response_queue.put(response)
            last_observation_time = time.time() # Reset timer after user interaction
        elif time.time() - last_observation_time > observation_interval:
            # Proactive observation
            response = generate_ai_response(latest_emotion, latest_eye_contact, latest_smiling)
            response_queue.put(response)
            last_observation_time = time.time()

        time.sleep(0.05) # Small sleep to prevent busy-waiting and allow other threads to run

    print("AI processing stopped.")

# ----------------------------
# Tkinter GUI Setup
# ----------------------------
def send_message():
    user_message = user_input.get()
    if user_message.strip():
        chat_history.config(state=tk.NORMAL)
        chat_history.insert(tk.END, "You: " + user_message + "\n", "user")
        chat_history.config(state=tk.DISABLED)
        question_queue.put(user_message) # Send question to AI thread
        user_input.delete(0, tk.END) # Clear input field

def update_chat_history():
    while not response_queue.empty():
        ai_response = response_queue.get()
        chat_history.config(state=tk.NORMAL)
        chat_history.insert(tk.END, "AI: " + ai_response + "\n", "ai")
        chat_history.config(state=tk.DISABLED)
        chat_history.yview(tk.END) # Auto-scroll to the bottom
    root.after(100, update_chat_history) # Check for new messages every 100ms

def on_closing():
    stop_video_flag.set() # Signal both threads to stop
    root.destroy()
    print("Application closed.")

# Main Tkinter window
root = tk.Tk()
root.title("AI Face Companion Chat")
root.geometry("500x400") # Adjust size as needed

# Chat history display
chat_history = scrolledtext.ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED, font=("Arial", 10))
chat_history.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Configure tags for different message colors
chat_history.tag_config("user", foreground="blue")
chat_history.tag_config("ai", foreground="green")

# User input field
user_input = Entry(root, font=("Arial", 12))
user_input.pack(padx=10, pady=5, fill=tk.X)
user_input.bind("<Return>", lambda event=None: send_message()) # Allow Enter key to send message
user_input.focus_set() # Ensure focus is set on the input field

# Send button
send_button = Button(root, text="Send", command=send_message, font=("Arial", 10))
send_button.pack(padx=10, pady=5)

# Start the video processing and AI processing in separate threads
video_thread = threading.Thread(target=video_processing_thread)
ai_thread = threading.Thread(target=ai_processing_thread)

video_thread.start()
ai_thread.start()

# Start periodically checking for AI responses
root.after(100, update_chat_history)

# Set protocol for closing the window
root.protocol("WM_DELETE_WINDOW", on_closing)

# Run the Tkinter event loop
root.mainloop()

# Ensure threads are joined upon shutdown (optional, but good practice)
video_thread.join()
ai_thread.join()
