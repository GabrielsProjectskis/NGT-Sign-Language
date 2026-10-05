"""
NGT Sign Language Recognition - Streamlit Web UI
A user-friendly interface for non-technical users.
Using the NEW MediaPipe Tasks API (compatible with mediapipe 0.10.30+)

Run with: streamlit run ui.py
"""

import streamlit as st
import cv2
import numpy as np
import json
import os
import urllib.request
from collections import deque
import time

# Import the new MediaPipe Tasks API
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Page config
st.set_page_config(
    page_title="NGT Sign Language Recognition",
    page_icon="🤟",
    layout="wide"
)

# Constants
DATA_DIR = "training_data"
SEQUENCE_LENGTH = 30
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


# ============ Helper Functions ============

def download_model():
    """Download the hand landmarker model if not present."""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading hand landmarker model..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        st.success("Model downloaded!")


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
    landmarks = []
    for lm in hand_landmarks:
        landmarks.extend([lm.x, lm.y, lm.z])
    return np.array(landmarks)


def normalize_landmarks(landmarks):
    landmarks = landmarks.reshape(-1, 3)
    wrist = landmarks[0]
    landmarks = landmarks - wrist
    scale = np.linalg.norm(landmarks[9])
    if scale > 0:
        landmarks = landmarks / scale
    return landmarks.flatten()


def load_training_data():
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
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    filepath = os.path.join(DATA_DIR, f"{letter.lower()}.json")
    
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
    else:
        data = {"samples": [], "is_dynamic": is_dynamic}
    
    if is_dynamic:
        data["samples"].append(landmarks_list)
    else:
        data["samples"].append(landmarks_list[-1] if isinstance(landmarks_list, list) else landmarks_list)
    
    data["is_dynamic"] = is_dynamic
    
    with open(filepath, 'w') as f:
        json.dump(data, f)
    
    return len(data['samples'])


def calculate_distance(landmarks1, landmarks2):
    return np.linalg.norm(np.array(landmarks1) - np.array(landmarks2))


def predict_letter(current_landmarks, training_data):
    best_letter = None
    best_distance = float('inf')
    
    for letter, data in training_data.items():
        if data.get("is_dynamic", False):
            continue
        
        samples = data["samples"]
        if not samples:
            continue
        
        distances = [calculate_distance(current_landmarks, sample) for sample in samples]
        avg_distance = np.mean(distances)
        
        if avg_distance < best_distance:
            best_distance = avg_distance
            best_letter = letter
    
    confidence = max(0, 1 - (best_distance / 2))
    
    if best_distance < 0.5:
        return best_letter, confidence
    return None, 0


@st.cache_resource
def get_detector():
    """Create and cache the hand landmarker detector."""
    download_model()
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    return vision.HandLandmarker.create_from_options(options)


# ============ Streamlit App ============

def main():
    st.title("🤟 NGT Sign Language Recognition")
    st.markdown("Learn and practice Dutch Sign Language (NGT) fingerspelling!")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    mode = st.sidebar.radio("Mode", ["🎯 Practice", "📝 Record New Signs", "📚 View Training Data"])
    
    # Load training data
    training_data = load_training_data()
    
    if training_data:
        st.sidebar.success(f"✅ Trained letters: {', '.join(sorted(training_data.keys()))}")
    else:
        st.sidebar.warning("⚠️ No training data yet. Record some signs first!")
    
    # Reference videos
    with st.sidebar.expander("📺 NGT Reference Videos"):
        st.markdown("""
        Learn correct signs from these videos:
        - [Video 1](https://www.youtube.com/watch?v=GMi9qDSw2o8)
        - [Video 2](https://www.youtube.com/watch?v=V06A3Sy9Lic)
        - [Video 3](https://www.youtube.com/watch?v=oZMyER7fWJY)
        - [Video 4](https://www.youtube.com/watch?v=Boo7pMze5rk)
        """)
    
    # Main content based on mode
    if mode == "🎯 Practice":
        practice_mode(training_data)
    elif mode == "📝 Record New Signs":
        record_mode()
    else:
        view_data_mode(training_data)


def practice_mode(training_data):
    st.header("Practice Mode")
    
    if not training_data:
        st.error("No training data available! Please record some signs first.")
        return
    
    st.info("Show a sign to the camera and see if it's recognized correctly!")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        run = st.checkbox("▶️ Start Camera")
        FRAME_WINDOW = st.image([])
    
    with col2:
        prediction_placeholder = st.empty()
        confidence_placeholder = st.empty()
        feedback_placeholder = st.empty()
    
    if run:
        detector = get_detector()
        cap = cv2.VideoCapture(0)
        
        while run:
            success, frame = cap.read()
            if not success:
                st.error("Cannot access camera!")
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image and detect
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = detector.detect(mp_image)
            
            if detection_result.hand_landmarks:
                # Draw landmarks
                annotated_frame = draw_landmarks_on_image(rgb_frame, detection_result)
                
                # Extract and normalize landmarks
                hand_landmarks = detection_result.hand_landmarks[0]
                raw_landmarks = extract_landmarks(hand_landmarks)
                current_landmarks = normalize_landmarks(raw_landmarks).tolist()
                
                letter, confidence = predict_letter(current_landmarks, training_data)
                
                if letter and confidence > 0.3:
                    prediction_placeholder.markdown(f"## 🔤 {letter}")
                    confidence_placeholder.progress(confidence)
                    
                    if confidence > 0.7:
                        feedback_placeholder.success("✅ Great sign!")
                    elif confidence > 0.5:
                        feedback_placeholder.warning("🤔 Almost there...")
                    else:
                        feedback_placeholder.info("Keep trying!")
                else:
                    prediction_placeholder.markdown("## ❓")
                    confidence_placeholder.empty()
                    feedback_placeholder.info("Show a sign...")
                
                FRAME_WINDOW.image(annotated_frame)
            else:
                FRAME_WINDOW.image(rgb_frame)
                prediction_placeholder.markdown("## 👋")
                feedback_placeholder.info("Show your hand to the camera...")
            
            time.sleep(0.03)
        
        cap.release()


def record_mode():
    st.header("Record New Signs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        letter = st.selectbox(
            "Select letter to record",
            list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
            key="record_letter"
        )
    
    with col2:
        is_dynamic = st.checkbox("Dynamic sign (involves movement)", value=False)
    
    st.markdown(f"### Recording: **{letter}**")
    
    if is_dynamic:
        st.info("🔄 Dynamic mode: Hold the sign and move as needed. Recording will capture the motion.")
    else:
        st.info("📸 Static mode: Hold your hand still in the correct position.")
    
    # Recording state
    if 'recording_frames' not in st.session_state:
        st.session_state.recording_frames = []
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_recording = st.button("🔴 Start Recording", disabled=st.session_state.is_recording)
    with col2:
        stop_recording = st.button("⏹️ Stop Recording", disabled=not st.session_state.is_recording)
    with col3:
        save_btn = st.button("💾 Save Recording", disabled=len(st.session_state.recording_frames) == 0)
    
    if start_recording:
        st.session_state.is_recording = True
        st.session_state.recording_frames = []
    
    if stop_recording:
        st.session_state.is_recording = False
    
    if save_btn and st.session_state.recording_frames:
        num_samples = save_letter_data(letter, st.session_state.recording_frames, is_dynamic)
        st.success(f"✅ Saved! Total samples for '{letter}': {num_samples}")
        st.session_state.recording_frames = []
    
    # Camera feed
    run_camera = st.checkbox("📷 Show Camera")
    FRAME_WINDOW = st.image([])
    status_placeholder = st.empty()
    
    if run_camera:
        detector = get_detector()
        cap = cv2.VideoCapture(0)
        
        while run_camera:
            success, frame = cap.read()
            if not success:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image and detect
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = detector.detect(mp_image)
            
            if detection_result.hand_landmarks:
                # Draw landmarks
                annotated_frame = draw_landmarks_on_image(rgb_frame, detection_result)
                
                if st.session_state.is_recording:
                    hand_landmarks = detection_result.hand_landmarks[0]
                    raw_landmarks = extract_landmarks(hand_landmarks)
                    current_landmarks = normalize_landmarks(raw_landmarks).tolist()
                    st.session_state.recording_frames.append(current_landmarks)
                
                # Recording indicator
                if st.session_state.is_recording:
                    cv2.circle(annotated_frame, (50, 50), 20, (255, 0, 0), -1)
                    cv2.putText(annotated_frame, f"REC: {len(st.session_state.recording_frames)} frames", 
                               (80, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                FRAME_WINDOW.image(annotated_frame)
            else:
                FRAME_WINDOW.image(rgb_frame)
            
            status_placeholder.text(f"Recorded frames: {len(st.session_state.recording_frames)}")
            
            time.sleep(0.03)
        
        cap.release()


def view_data_mode(training_data):
    st.header("Training Data Overview")
    
    if not training_data:
        st.warning("No training data yet!")
        return
    
    # Summary table
    data_summary = []
    for letter, data in sorted(training_data.items()):
        data_summary.append({
            "Letter": letter,
            "Samples": len(data["samples"]),
            "Type": "Dynamic" if data.get("is_dynamic", False) else "Static"
        })
    
    st.dataframe(data_summary, use_container_width=True)
    
    # Delete option
    st.subheader("Manage Data")
    
    letter_to_delete = st.selectbox(
        "Select letter to delete",
        [""] + list(training_data.keys())
    )
    
    if letter_to_delete and st.button(f"🗑️ Delete all data for '{letter_to_delete}'"):
        filepath = os.path.join(DATA_DIR, f"{letter_to_delete.lower()}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
            st.success(f"Deleted data for '{letter_to_delete}'")
            st.rerun()


if __name__ == "__main__":
    main()
