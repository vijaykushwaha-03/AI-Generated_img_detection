# Dual-Stream Frequency-Spatial Network Architecture

## Overview
This project utilizes a custom **Dual-Stream Architecture** designed specifically for AI image detection. Unlike standard CNNs that focus solely on semantic content, this network explicitly separates "visual semantics" from "signal artifacts" (noise residuals), making it highly robust against deepfakes and generative AI outputs.

## Architecture Diagram (Conceptual)
```mermaid
graph TD
    Input[Input Image (RGB)] --> Split{Split Streams}
    
    %% Stream 1: RGB
    Split --> RGB_Stream[RGB Stream]
    RGB_Stream --> EffNet1[EfficientNet-B0 Backbone]
    EffNet1 --> Feat1[Semantic Features (1280d)]
    
    %% Stream 2: Noise
    Split --> Noise_Stream[Noise Stream]
    Noise_Stream --> SRM[SRM Constrained Filters]
    SRM --> Residuals[Noise Residuals (9ch)]
    Residuals --> Adapt[1x1 Conv (9->3ch)]
    Adapt --> EffNet2[EfficientNet-B0 Backbone]
    EffNet2 --> Feat2[Signal Features (1280d)]
    
    %% Fusion
    Feat1 --> Concat[Concatenation]
    Feat2 --> Concat
    Concat --> Fusion[Fusion Layer (2560d)]
    Fusion --> Dense[Dense Layer (256)]
    Dense --> Classifier[Final Binary Classifier]
```

## Component Details

### 1. RGB Stream (Semantic Analysis)
-   **Purpose**: Detects high-level semantic anomalies typical of AI generation (e.g., asymmetrical eyes, impossible geometry, lighting inconsistencies).
-   **Backbone**: `EfficientNet-B0` (Pretrained on ImageNet).
-   **Input**: Standard RGB image (Normalized).

### 2. Noise Stream (Signal Analysis)
-   **Purpose**: Detects invisible high-frequency artifacts left by upsampling layers (checkerboard artifacts) and diffusion noise schedules.
-   **Input Layer**: **SRM (Spatial Rich Model) Filters**.
    -   These are 3 fixed, non-learnable 5x5 kernels derived from digital forensics (Steganalysis).
    -   They act as high-pass filters to suppress image content and isolate the "noise residual."
-   **Feature Extractor**: A secondary `EfficientNet-B0` that learns statistical patterns within the noise domain.

### 3. Fusion & Classification
-   **Strategy**: Late Fusion.
-   **Mechanism**: The 1280-dimensional feature vectors from both streams are concatenated (Total: 2560 dimensions).
-   **Head**: A fully connected block `(Linear -> ReLU -> Dropout -> Linear)` maps the fused features to the 2 output classes: `Real` vs `AI-Generated`.

## Why This Works
| Feature | Standard CNN | Dual-Stream Network |
| :--- | :--- | :--- |
| **Focus** | "What is in the image?" | "How was the image created?" |
| **Noise Handling** | Pools/Downsamples noise away (bad for detection). | Explicitly amplifies noise via SRM filters. |
| **Robustness** | Can be fooled by semantically perfect AI images. | Detects the invisible statistical trace of the generator. |
