# 🕵️ AI vs Real Image Detector (Dual-Stream Architecture)

A robust, "interview-ready" computer vision project that detects AI-generated images by analyzing both **semantic content** and **invisible frequency artifacts**.

## 🚀 The Problem
Standard CNNs (ResNet, EfficientNet) often fail to detect modern AI images because they are designed to be invariant to noise and high-frequency details (due to pooling and downsampling). However, AI generators (GANs, Diffusion models) leave behind distinct "fingerprints" in the noise residual domain (e.g., checkerboard artifacts, abnormal spectral peaks) that human eyes—and standard CNNs—often miss.

## 🧠 How We Achieve High Accuracy (The Dual-Stream Approach)

Our model effectively acts as two experts working together:

### 1. The "Visual" Expert (RGB Stream)
*   **What it sees:** The actual image (faces, objects, lighting).
*   **How it helps:** It detects semantic errors common in AI generation, such as:
    *   Asymmetrical eyes or strange pupil shapes.
    *   Hands with too many/few fingers.
    *   Impossible lighting or shadows.
    *   "Smoothness" typical of plastic-looking AI skin.

### 2. The "Forensic" Expert (Noise Stream)
*   **What it sees:** **Invisible noise patterns** hidden in the pixels.
*   **How it works:** We use **SRM (Spatial Rich Model) Filters**, a technique from digital forensics. These filters subtract the image content to reveal the "residue" or "fingerprint" left by the camera sensor or the AI generator.
    *   *Real Cameras* leave consistent, random sensor noise (ISO grain).
    *   *AI Generators* (GANs/Diffusion) leave structured, checkerboard patterns or abnormal high-frequency grids due to upsampling.

### 🏆 Project Performance (The "Score")

#### Model Accuracy
By combining these two streams, we aim for **>95% detection accuracy**.
*   **Standard Models (ResNet)**: Often fail (~60-70% accuracy) because they treat the noise as "unimportant" and filter it out.
*   **Our Model**: Explicitly hunts for this noise, allowing it to catch even "perfect-looking" AI images that fool humans.

#### Interpreting the Confidence Score
When you use the app, you see a **Confidence Score** (0% - 100%).
*   **90-100%**: The model is extremely sure. It likely found both visual artifacts *and* forensic noise evidence.
*   **60-70%**: The model is uncertain. The image might be heavily compressed (destroying noise) or highly realistic.
*   **Note**: This score is a probability (Softmax), not a guarantee.

## 🛠️ Project Structure
```
ai_image_detector/
├── data/               # (User must populate) train/ and val/ folders
│   ├── train/          # Class folders: 'real' and 'ai'
│   └── val/
├── model/
│   ├── architecture.py # Custom Dual-Stream PyTorch Model
│   └── layers.py       # SRM Filter implementation
├── models/             # Saved checkpoints (.pth)
├── train.py            # Training loop with Albumentations & Early Stopping
├── inference.py        # Prediction logic with confidence scoring
├── app.py              # Streamlit Web UI
└── requirements.txt    # Project dependencies
```

## 💻 Tech Stack
-   **PyTorch**: Core Deep Learning framework.
-   **Albumentations**: Advanced data augmentation (essential for simulating JPEG compression to make the model robust).
-   **Streamlit**: For the demo interface.
-   **TIMM**: for efficient backbone implementations.

## 🏃‍♂️ How to Run

### 1. Setup
Create a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r ai_image_detector/requirements.txt
```

### 2. Prepare Data
Put your dataset in `ai_image_detector/data/`. Structure:
```
data/
  train/
    real/   # camera photos
    ai/     # midjourney/dalle images
  val/
    real/
    ai/
```

### 3. Train
```bash
python ai_image_detector/train.py --epochs 10 --batch_size 32
```
This saves the best model to `ai_image_detector/models/ai_image_detector.pth`.

### 4. Run UI
```bash
streamlit run ai_image_detector/app.py
```
Upload an image to see the Dual-Stream network in action.

## ⚠️ Limitations & Future Work
-   **Compression Attacks**: While I used JPEG compression augmentation, extremely heavy compression can destroy the high-frequency signal used by the SRM stream.
-   **Generator Evolution**: New models (Midjourney v6, Flux) have different artifacts. Continuous retraining is required.
-   **False Positives**: Heavily filtered "real" photos (Snapchat filters) might trigger the noise stream.
-   **Future**: Add a Frequency Domain branch (FFT) to explicitly detect spectral grid artifacts.

---
**Why this over a generic model?**
By explicitly separating the "Signal" from the "Semantics," this model is more interpretable and robust against high-quality generations that look semantically perfect.
