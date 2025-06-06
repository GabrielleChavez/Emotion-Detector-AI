from deepface import DeepFace
import cv2 # OpenCV for loading/displaying image (optional display)
import matplotlib.pyplot as plt # For displaying image in some environments

# Path to the image you want to analyze
image_path = "angry_child.jpg" # <--- CHANGE THIS TO YOUR IMAGE PATH

try:
    # Load the image (optional, just for display later if you want)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        exit()

    # Analyze the image for emotions
    # DeepFace.analyze can perform multiple actions: emotion, age, gender, race
    # We are interested in 'emotion'
    # enforce_detection=False means it won't raise an error if no face is found,
    # but will return an empty result or less complete data.
    # For emotion, it's best to ensure a face is detected.
    # You can also specify detector_backend: 'opencv', 'ssd', 'mtcnn', 'retinaface', 'dlib'
    analysis_results = DeepFace.analyze(
        img_path=image_path,
        actions=['emotion'],
        enforce_detection=True, # Set to False if you don't want an error on no face
        detector_backend='opencv' # Using opencv as a common backend
    )

    # The result is a list of dictionaries, one for each detected face.
    # For simplicity, we'll assume one face or process the first detected face.
    if analysis_results and len(analysis_results) > 0:
        first_face_analysis = analysis_results[0]
        dominant_emotion = first_face_analysis['dominant_emotion']
        emotions = first_face_analysis['emotion'] # Dictionary of all emotion scores

        print(f"Image: {image_path}")
        print(f"Dominant Emotion: {dominant_emotion}")
        print("Emotion Scores:")
        for emotion, score in emotions.items():
            print(f"  - {emotion}: {score:.2f}%")

        # Optional: Display the image with the dominant emotion
        # (matplotlib is good for scripts/notebooks)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Dominant Emotion: {dominant_emotion}")
        plt.axis('off') # Hide axes
        plt.show()

    else:
        print(f"No face detected in {image_path} or analysis failed.")

except ValueError as ve:
    # This specific exception is often raised by DeepFace if no face is detected
    # when enforce_detection=True
    print(f"ValueError: {ve}")
    print("This likely means no face was detected in the image.")
except Exception as e:
    print(f"An error occurred: {e}")
    print("Make sure the image path is correct and the image contains a detectable face.")
    print("Also, the first time you run it, DeepFace might download pre-trained models, so ensure you have an internet connection.")