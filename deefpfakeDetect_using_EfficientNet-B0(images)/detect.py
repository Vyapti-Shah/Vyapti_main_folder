import argparse
import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image

def load_model(model_path, device):
    """Loads the EfficientNet-B0 model with custom weights."""
    print("Loading EfficientNet-B0 model...")
    # Initialize the standard EfficientNet-B0 architecture
    # Use weights=None for newer torchvision, pretrained=False for older
    try:
        model = models.efficientnet_b0(weights=None)
    except TypeError:
        model = models.efficientnet_b0(pretrained=False)
    
    # Modify the classifier for binary classification (Real vs AI Generated)
    # Assuming the model was trained with 2 output classes
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)
    
    # Load the weights
    try:
        state_dict = torch.load(model_path, map_location=device)
        # Handle case where state_dict might be nested (e.g., inside 'state_dict' or 'model')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
            
        model.load_state_dict(state_dict, strict=False)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Ensure the .pt file matches the EfficientNet-B0 architecture with 2 output classes.")
        exit(1)
        
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    """Loads and preprocesses the image for EfficientNet."""
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        exit(1)
        
    # Standard EfficientNet preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img_tensor = transform(img).unsqueeze(0) # Add batch dimension
    return img_tensor

def add_watermark(image_path, output_path, is_real):
    """Adds a text watermark to the top-left corner of the image."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error reading image with OpenCV: {image_path}")
        return

    # Text configuration
    text = "REAL" if is_real else "AI Generated"
    # OpenCV uses BGR color format
    color = (0, 255, 0) if is_real else (0, 0, 255) # Green for REAL, Red for AI Generated
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    
    # Get text size to draw a background rectangle for better visibility
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = 10, 30 # Top-left position
    
    # Draw black background rectangle for text visibility
    cv2.rectangle(img, (x, y - text_height - 5), (x + text_width, y + baseline), (0, 0, 0), cv2.FILLED)
    
    # Draw text
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    
    cv2.imwrite(output_path, img)
    print(f"Watermarked image saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="AI Image Detector using EfficientNet-B0")
    parser.add_argument("-i", "--input", required=True, help="Path to input image")
    parser.add_argument("-o", "--output", required=True, help="Path to save watermarked image")
    parser.add_argument("-m", "--model", required=True, help="Path to model weights (.pt or .pth)")
    parser.add_argument("--fake_class_index", type=int, default=1, help="Index of the 'Fake/AI' class in model output (default: 1)")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = load_model(args.model, device)
    img_tensor = preprocess_image(args.input).to(device)
    
    print("Classifying image...")
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        
    print(f"Output Probabilities: {probabilities[0].cpu().numpy()}")
    
    is_fake = (predicted_class == args.fake_class_index)
    
    if is_fake:
        print("Prediction: AI Generated / Fake")
    else:
        print("Prediction: Real")
        
    add_watermark(args.input, args.output, not is_fake)

if __name__ == "__main__":
    main()
