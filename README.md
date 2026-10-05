# 🤟 NGT Sign Language Fingerspelling Recognition

A real-time Dutch Sign Language (NGT) fingerspelling recognition system using MediaPipe hand landmarks and simple classification.

## Overview

This project provides:
- **Real-time hand landmark detection** using MediaPipe
- **Custom training** - record your own sign samples
- **Two interfaces**: OpenCV (terminal) and Streamlit (web UI)
- **Support for both static and dynamic letters**

### Approach

We use **MediaPipe Hand Landmarks** to extract 21 hand keypoints (63 values: x, y, z for each point). These landmarks are normalized relative to the wrist position and hand size, making recognition invariant to:
- Hand position in frame
- Distance from camera
- Hand size

For **static letters**: Simple nearest-neighbor matching against recorded templates.

For **dynamic letters**: Sequence comparison using FastDTW (Dynamic Time Warping) or averaged landmark positions.

**Why this approach?**
- No large dataset required - you create your own samples
- Works in real-time
- Easy to understand and extend
- MediaPipe handles the hard part (hand detection)

---

## Setup

### 1. Prerequisites

- Python 3.8 or higher
- Webcam
- (Optional) Virtual environment

### 2. Installation

```bash
# Clone or download this project
cd ngt_sign_language

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -c "import cv2; import mediapipe; print('Ready!')"
```

---

## Usage

### Option 1: OpenCV Interface (Terminal)

```bash
python app.py
```

**Controls:**
| Key | Action |
|-----|--------|
| `R` | Start/stop recording mode |
| `D` | Toggle dynamic letter mode |
| `A-Z` | Set letter to record (while in recording mode) |
| `SPACE` | Save current recording |
| `C` | Clear current recording |
| `Q` | Quit |

**Workflow:**
1. Press `R` to enter recording mode
2. Press a letter key (e.g., `A`) to set which letter you're recording
3. Make the sign with your hand
4. Press `SPACE` to save
5. Repeat for more samples (10-20 per letter recommended)
6. Press `R` to exit recording and test recognition

### Option 2: Streamlit Web UI (Recommended for non-technical users)

```bash
streamlit run ui.py
```

This opens a browser with a user-friendly interface featuring:
- **Practice Mode**: Test your signs against trained letters
- **Record Mode**: Add new training samples
- **View Data**: See and manage your training data

---

## Creating Training Data

### Static Letters (most letters)

1. Watch the [reference videos](#reference-videos) to learn correct hand positions
2. Enter recording mode
3. Hold your hand steady in the correct position
4. Record 10-20 samples per letter from slightly different angles

### Dynamic Letters (J, Z, and others with movement)

1. Enable "Dynamic" mode (`D` key or checkbox)
2. Perform the full motion of the sign
3. The system records the sequence of movements

### Tips for Good Training Data

- **Lighting**: Ensure good, consistent lighting
- **Background**: Plain backgrounds work best
- **Variety**: Record samples with slight variations in angle and position
- **Consistency**: Use the same hand (preferably dominant hand)
- **Reference**: Always check the official NGT videos for correct form

---

## Reference Videos

Learn correct NGT fingerspelling from these official resources:

- [Video 1](https://www.youtube.com/watch?v=GMi9qDSw2o8)
- [Video 2](https://www.youtube.com/watch?v=V06A3Sy9Lic)
- [Video 3](https://www.youtube.com/watch?v=oZMyER7fWJY)
- [Video 4](https://www.youtube.com/watch?v=Boo7pMze5rk)

---

## Project Structure

```
ngt_sign_language/
├── app.py              # Main OpenCV application
├── ui.py               # Streamlit web interface
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── training_data/      # Saved letter samples (created automatically)
    ├── a.json
    ├── b.json
    └── ...
```

---

## How It Works

### 1. Hand Detection (MediaPipe Tasks API)

This project uses the **new MediaPipe Tasks API** (required for mediapipe 0.10.30+). The model file (`hand_landmarker.task`) is automatically downloaded on first run.

MediaPipe detects the hand and returns 21 landmarks:

```
        8   12  16  20
        |   |   |   |
    4   7   11  15  19
    |   |   |   |   |
    3   6   10  14  18
    |   |   |   |   |
    2   5   9   13  17
    |    \  |  /   /
    1      \|/    /
            0----
          (wrist)
```

### 2. Normalization

Landmarks are normalized to be:
- **Position-invariant**: Centered on wrist (point 0)
- **Scale-invariant**: Scaled by palm size (wrist to middle finger base)

### 3. Classification

**Static**: Euclidean distance to stored templates
**Dynamic**: FastDTW sequence matching (or averaged comparison)

---

## Future Considerations

### Potential Improvements

1. **Deep Learning Classifier**
   - Train a neural network on landmarks for better accuracy
   - Use LSTM/GRU for dynamic letter sequences

2. **Data Augmentation**
   - Add noise to training samples
   - Synthetic rotation/scaling

3. **Word/Sentence Recognition**
   - Chain letter predictions into words
   - Add language model for correction

4. **Two-Hand Support**
   - Some NGT signs use both hands
   - MediaPipe can track multiple hands

5. **Mobile App**
   - Port to mobile using MediaPipe's mobile solutions

### Known Limitations

- Requires good lighting
- Single hand only (current implementation)
- Needs ~10+ samples per letter for reliable recognition
- Some similar letters may be confused (M/N, U/V)

---

## Troubleshooting

**Camera not working?**
- Check if another app is using the camera
- Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`

**Low accuracy?**
- Record more training samples (20+ per letter)
- Ensure consistent lighting
- Check reference videos for correct hand positions

**Streamlit not loading?**
- Make sure you're running `streamlit run ui.py` not `python ui.py`
- Check that port 8501 is not in use

---

## Credits

- [MediaPipe](https://mediapipe.dev/) - Hand landmark detection
- [Hand Gesture Recognition MediaPipe](https://github.com/kinivi/hand-gesture-recognition-mediapipe) - Inspiration

---

## License

This project is for educational purposes as part of the BUaS ADS&AI challenge.
