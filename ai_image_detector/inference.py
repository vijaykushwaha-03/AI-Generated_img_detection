import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
from model.architecture import DualStreamDetector

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_inference_transforms(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def load_model(model_path="ai_image_detector/models/ai_image_detector.pth"):
    # Initialize model architecture
    model = DualStreamDetector(pretrained=False) # No need to download weights logic again if we load state_dict
    # However, if we rely on timm's pretrained initialization for structure, we might want True or ensure dimensions match.
    # Since we save the state dict of the whole model (including backbone), pretrained=False is safer for strict loading 
    # if we saved it fully. But usually we define the structure first.
    # If the user runs inference without training, we need a fallback or pretrained=True just to have a working model (untrained).
    
    try:
        if torch.cuda.is_available():
            state_dict = torch.load(model_path)
        else:
            state_dict = torch.load(model_path, map_location='cpu')
        
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print(f"Warning: Model checkpoint not found at {model_path}. Using random weights.")
    except Exception as e:
        print(f"Error loading model: {e}")
    
    model.to(device)
    model.eval()
    return model

def predict(model, image_path_or_obj):
    """
    Predicts if an image is Real or AI.
    args:
        model: loaded pytorch model
        image_path_or_obj: str (path) or PIL.Image or numpy array
    returns:
        dict: {label: str, confidence: float, class_id: int}
    """
    transform = get_inference_transforms()
    
    # helper to load image
    if isinstance(image_path_or_obj, str):
        image = np.array(Image.open(image_path_or_obj).convert("RGB"))
    elif isinstance(image_path_or_obj, Image.Image):
        image = np.array(image_path_or_obj.convert("RGB"))
    elif isinstance(image_path_or_obj, np.ndarray):
        image = image_path_or_obj
    else:
        raise ValueError("Input must be path, PIL Image, or numpy array")

    # Preprocess
    augmented = transform(image=image)
    img_tensor = augmented["image"].unsqueeze(0).to(device) # [1, 3, H, W]
    
    classes = ['Real', 'AI-Generated']
    
    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)
        
        conf, pred_idx = torch.max(probs, 1)
        
        confidence = conf.item()
        label = classes[pred_idx.item()]
        
    return {
        "label": label,
        "confidence": confidence,
        "class_id": pred_idx.item(),
        "probabilities": probs.cpu().numpy()[0]
    }

if __name__ == "__main__":
    # Test run
    model = load_model()
    # Create a dummy image for testing
    dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    result = predict(model, dummy_img)
    print("Test Prediction:", result)
