# 🕵️ AI vs Real Image Detector (Dual-Stream Architecture)

A robust, "interview-ready" computer vision project that detects AI-generated images by analyzing both **semantic content** and **invisible frequency artifacts**.

## 🚀 The Problem
Standard CNNs (ResNet, EfficientNet) often fail to detect modern AI images because they are designed to be invariant to noise and high-frequency details (due to pooling and downsampling). However, AI generators (GANs, Diffusion models) leave behind distinct "fingerprints" in the noise residual domain (e.g., checkerboard artifacts, abnormal spectral peaks) that human eyes—and standard CNNs—often miss.

## 🧠 The Solution: Frequency-Spatial Hybrid Network
I designed a custom **Dual-Stream Architecture** that mimics digital forensics techniques:

1.  **RGB Stream (Semantic)**:
    -   Uses an **EfficientNet-B0** backbone to analyze visual semantics (e.g., asymmetrical eyes, warped hands, impossible lighting).
2.  **Noise Stream (Signal)**:
    -   Starts with **SRM (Spatial Rich Model) Filters**. These are non-learnable high-pass filters used in steganalysis to extract noise residuals.
    -   This stream forces the model to ignore the *picture* and look at the *pixels*, revealing the generator's statistical trace.
3.  **Fusion Head**:
    -   Concatenates specific features from both streams to make a final decision with high confidence.

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
