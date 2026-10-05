"""
NGT Sign Language Fingerspelling Recognition
Main application with real-time webcam detection
Using the NEW MediaPipe Tasks API (compatible with mediapipe 0.10.30+)
"""

import cv2
import numpy as np
import os
import json
import urllib.request
from collections import deque

# Import the new MediaPipe Tasks API
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Constants
DATA_DIR = "training_data"
SEQUENCE_LENGTH = 30  # For dynamic letters (number of frames to consider)
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def download_model():
    """Download the hand landmarker model if not present."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded!")


def draw_landmarks_on_image(rgb_image, detection_result):
    """Draw hand landmarks on the image using OpenCV."""
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = np.copy(rgb_image)
    
    # Hand connections (pairs of landmark indices to connect)
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
        (5, 9), (9, 13), (13, 17)  # Palm
    ]
    
    h, w, _ = annotated_image.shape
    
    for hand_landmarks in hand_landmarks_list:
        # Draw connections
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start = hand_landmarks[start_idx]
            end = hand_landmarks[end_idx]
            
            start_point = (int(start.x * w), int(start.y * h))
            end_point = (int(end.x * w), int(end.y * h))
            
            cv2.line(annotated_image, start_point, end_point, (0, 255, 0), 2)
        
        # Draw landmarks
        for landmark in hand_landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(annotated_image, (x, y), 5, (255, 0, 0), -1)
    
    return annotated_image


def extract_landmarks(hand_landmarks):
    """
    Extract normalized landmark coordinates from hand landmarks.
    Returns a flattened array of 63 values (21 landmarks × 3 coordinates).
    """
    landmarks = []
    for lm in hand_landmarks:
        landmarks.extend([lm.x, lm.y, lm.z])
    return np.array(landmarks)


def normalize_landmarks(landmarks):
    """
    Normalize landmarks relative to wrist position and hand size.
    This makes the model invariant to hand position and scale.
    """
    landmarks = landmarks.reshape(-1, 3)
    
    # Use wrist (index 0) as origin
    wrist = landmarks[0]
    landmarks = landmarks - wrist
    
    # Scale by distance from wrist to middle finger MCP (index 9)
    scale = np.linalg.norm(landmarks[9])
    if scale > 0:
        landmarks = landmarks / scale
    
    return landmarks.flatten()


def load_training_data():
    """Load all saved training data for each letter."""
    training_data = {}
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        return training_data
    
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.json'):
            letter = filename.replace('.json', '').upper()
            filepath = os.path.join(DATA_DIR, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                training_data[letter] = data
    
    return training_data


def save_letter_data(letter, landmarks_list, is_dynamic=False):
    """Save recorded landmarks for a letter."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    filepath = os.path.join(DATA_DIR, f"{letter.lower()}.json")
    
    # Load existing data or create new
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
    else:
        data = {"samples": [], "is_dynamic": is_dynamic}
    
    # Add new sample
    if is_dynamic:
        # For dynamic letters, save the sequence
        data["samples"].append(landmarks_list)
    else:
        # For static letters, save single frame
        data["samples"].append(landmarks_list[-1] if isinstance(landmarks_list, list) else landmarks_list)
    
    data["is_dynamic"] = is_dynamic
    
    with open(filepath, 'w') as f:
        json.dump(data, f)
    
    print(f"Saved sample for letter '{letter}'. Total samples: {len(data['samples'])}")


def calculate_distance(landmarks1, landmarks2):
    """Calculate Euclidean distance between two landmark arrays."""
    return np.linalg.norm(np.array(landmarks1) - np.array(landmarks2))


def predict_static_letter(current_landmarks, training_data):
    """
    Predict letter using simple nearest neighbor matching.
    Returns the letter with minimum average distance to training samples.
    """
    best_letter = None
    best_distance = float('inf')
    
    for letter, data in training_data.items():
        if data.get("is_dynamic", False):
            continue  # Skip dynamic letters for static prediction
        
        samples = data["samples"]
        if not samples:
            continue
        
        # Calculate average distance to all samples of this letter
        distances = [calculate_distance(current_landmarks, sample) for sample in samples]
        avg_distance = np.mean(distances)
        
        if avg_distance < best_distance:
            best_distance = avg_distance
            best_letter = letter
    
    # Only return prediction if distance is below threshold
    confidence = max(0, 1 - (best_distance / 2))  # Simple confidence score
    
    if best_distance < 0.5:  # Threshold for accepting prediction
        return best_letter, confidence
    return None, 0


def predict_dynamic_letter(landmark_sequence, training_data):
    """
    Predict dynamic letter using FastDTW or simple sequence matching.
    For simplicity, we use averaged landmarks comparison here.
    For better results, install fastdtw: pip install fastdtw
    """
    try:
        from fastdtw import fastdtw
        use_dtw = True
    except ImportError:
        use_dtw = False
    
    best_letter = None
    best_distance = float('inf')
    
    # Average the sequence for simple comparison
    avg_landmarks = np.mean(landmark_sequence, axis=0)
    
    for letter, data in training_data.items():
        if not data.get("is_dynamic", False):
            continue  # Skip static letters
        
        samples = data["samples"]
        if not samples:
            continue
        
        for sample in samples:
            if use_dtw:
                # Use DTW for sequence comparison
                distance, _ = fastdtw(landmark_sequence, sample)
            else:
                # Simple: compare averaged landmarks
                sample_avg = np.mean(sample, axis=0)
                distance = calculate_distance(avg_landmarks, sample_avg)
            
            if distance < best_distance:
                best_distance = distance
                best_letter = letter
    
    confidence = max(0, 1 - (best_distance / 5))
    
    if best_distance < 2.0:
        return best_letter, confidence
    return None, 0


def main():
    """Main application loop."""
    print("\n" + "="*60)
    print("NGT Sign Language Fingerspelling Recognition")
    print("="*60)
    print("\nControls:")
    print("  R      - Start/stop RECORDING mode for a letter")
    print("  D      - Toggle DYNAMIC letter mode (for moving signs)")
    print("  SPACE  - Save current recording")
    print("  C      - Clear current recording")
    print("  Q      - Quit")
    print("\nIn recording mode, press the letter key (A-Z) to record.")
    print("="*60 + "\n")
    
    # Download model if needed
    download_model()
    
    # Load existing training data
    training_data = load_training_data()
    print(f"Loaded training data for letters: {list(training_data.keys())}")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # State variables
    recording = False
    is_dynamic = False
    current_letter = None
    recorded_landmarks = []
    landmark_buffer = deque(maxlen=SEQUENCE_LENGTH)
    
    # Setup MediaPipe Hand Landmarker with the new Tasks API
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    detector = vision.HandLandmarker.create_from_options(options)
    
    print("Hand landmarker initialized. Starting camera...")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read from webcam")
            break
        
        # Flip for selfie view
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect hand landmarks
        detection_result = detector.detect(mp_image)
        
        current_landmarks = None
        
        # Process detected hands
        if detection_result.hand_landmarks:
            # Draw landmarks on frame
            annotated_frame = draw_landmarks_on_image(rgb_frame, detection_result)
            frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            
            # Extract and normalize landmarks from first hand
            hand_landmarks = detection_result.hand_landmarks[0]
            raw_landmarks = extract_landmarks(hand_landmarks)
            current_landmarks = normalize_landmarks(raw_landmarks).tolist()
            
            # Add to buffer for dynamic letters
            landmark_buffer.append(current_landmarks)
        
        # Recording mode
        if recording and current_landmarks:
            recorded_landmarks.append(current_landmarks)
            cv2.circle(frame, (50, 50), 20, (0, 0, 255), -1)  # Red recording indicator
            cv2.putText(frame, f"Recording: {current_letter}", (80, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Frames: {len(recorded_landmarks)}", (80, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Prediction mode (when not recording)
        elif current_landmarks and training_data:
            # Try static prediction
            letter, confidence = predict_static_letter(current_landmarks, training_data)
            
            # Try dynamic prediction if we have enough frames
            if len(landmark_buffer) >= SEQUENCE_LENGTH:
                dyn_letter, dyn_conf = predict_dynamic_letter(
                    list(landmark_buffer), training_data
                )
                if dyn_conf > confidence:
                    letter, confidence = dyn_letter, dyn_conf
            
            if letter and confidence > 0.3:
                color = (0, 255, 0) if confidence > 0.6 else (0, 255, 255)
                cv2.putText(frame, f"Predicted: {letter}", (w - 300, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                cv2.putText(frame, f"Confidence: {confidence:.1%}", (w - 300, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Display mode indicator
        mode_text = "DYNAMIC" if is_dynamic else "STATIC"
        cv2.putText(frame, f"Mode: {mode_text}", (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show available letters
        if training_data:
            letters_text = "Trained: " + ", ".join(sorted(training_data.keys()))
            cv2.putText(frame, letters_text, (10, h - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Display frame
        cv2.imshow('NGT Sign Language Recognition', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            recording = not recording
            if recording:
                recorded_landmarks = []
                print("Recording started. Press a letter key (A-Z) to set the letter.")
            else:
                print("Recording stopped.")
        elif key == ord('d'):
            is_dynamic = not is_dynamic
            print(f"Dynamic mode: {is_dynamic}")
        elif key == ord(' ') and recorded_landmarks and current_letter:
            # Save recording
            save_letter_data(current_letter, recorded_landmarks, is_dynamic)
            training_data = load_training_data()  # Reload
            recorded_landmarks = []
            recording = False
            current_letter = None
        elif key == ord('c'):
            # Clear recording
            recorded_landmarks = []
            print("Recording cleared.")
        elif recording and ord('a') <= key <= ord('z'):
            # Set current letter
            current_letter = chr(key).upper()
            print(f"Recording for letter: {current_letter}")
    
    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()
