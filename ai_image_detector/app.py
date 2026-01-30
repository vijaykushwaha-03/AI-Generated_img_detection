import streamlit as st
import numpy as np
from PIL import Image
from inference import load_model, predict
import os

# Page Config
st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🕵️",
    layout="centered"
)

# Title & Description
st.title("🕵️ AI vs Real Image Detector")
st.markdown("""
This tool uses a **Dual-Stream Frequency-Spatial Network** to detect AI-generated images.
It analyzes both visual content and invisible high-frequency artifacts (noise residuals).

**Upload an image to check if it's Camera-Generated (Real) or AI-Generated.**
""")

# Load Model (Cached)
@st.cache_resource
def get_model():
    # Attempt to find the model path
    path = "ai_image_detector/models/ai_image_detector.pth"
    if not os.path.exists(path):
        # Fallback for running from different directories
        path = "models/ai_image_detector.pth"
    return load_model(path)

try:
    model = get_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png', 'webp'])

if uploaded_file is not None:
    try:
        # Display Image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Predict Button
        if st.button('Analyze Image'):
            with st.spinner('Analyzing pixel artifacts...'):
                result = predict(model, image)
                
                # Display Results
                label = result['label']
                conf = result['confidence']
                
                st.divider()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Prediction", label)
                
                with col2:
                    st.metric("Model Confidence", f"{conf:.2%}")
                
                # Contextual Message
                if label == "AI-Generated":
                    st.warning("⚠️ The model detected high-frequency artifacts consistent with generative AI.")
                else:
                    st.success("✅ The model analyzed the noise patterns and identified them as consistent with natural camera sensors.")
                
                st.info("Note: Confidence reflects the model's internal probability, not absolute certainty. Always verify with other methods.")
                
                # Debug info (optional)
                with st.expander("Detailed Probabilities"):
                    st.write(f"Real: {result['probabilities'][0]:.4f}")
                    st.write(f"AI: {result['probabilities'][1]:.4f}")

    except Exception as e:
        st.error(f"Error processing image: {e}")

st.divider()
st.caption("Architecture: EfficientNet-B0 + SRM Noise Stream | Framework: PyTorch")
