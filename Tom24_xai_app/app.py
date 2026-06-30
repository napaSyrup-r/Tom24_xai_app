import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import resnet18, densenet121, squeezenet1_1, efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
import zipfile
import os
from pathlib import Path
import random

import os
from huggingface_hub import hf_hub_download

HF_REPO_ID = "napaSyrup/Tom24_xai_weights"

REQUIRED_WEIGHTS = [
    "custom_cnn_best.pth",
    "resnet18_best.pth",
    "efficientnet_best.pth",
    "densenet121_best.pth",
    "squeezenet_best.pth"
]

print("Checking for model weights...")
for file_name in REQUIRED_WEIGHTS:
    if not os.path.exists(file_name):
        print(f"{file_name} missing. Downloading from Hugging Face Hub...")
        hf_hub_download(repo_id=HF_REPO_ID, filename=file_name, local_dir=".")
        print(f"Successfully downloaded {file_name}!")

        
# XAI imports


import lime
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM, AblationCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# Custom CNN Architecture
class LeafNetCNN(nn.Module):
    def __init__(self, num_classes):
        super(LeafNetCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=1)
        self.dropout1 = nn.Dropout(0.3)  
        self.dropout2 = nn.Dropout(0.5)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        self.pool = nn.MaxPool2d(2, 2)
   
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout1(x)  
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = F.relu(self.conv5(x))
        x = self.dropout2(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Configuration
#CLASS_NAMES = ['maize_diseases', 'maize_pests', 'maize_pests_activities',
#               'onion_diseases', 'onion_pests', 'tomato_diseases', 'tomato_pests']
#NUM_CLASSES = len(CLASS_NAMES)
#IMAGE_SIZE = 224


# Configuration
# ✅ MODIFIED: Add assertion to ensure CLASS_NAMES is not empty
CLASS_NAMES = ['maize_diseases', 'maize_pests', 'maize_pests_activities',
               'onion_diseases', 'onion_pests', 'tomato_diseases', 'tomato_pests']

# ✅ ADD: Validate that CLASS_NAMES is not empty
if len(CLASS_NAMES) == 0:
    raise ValueError("CLASS_NAMES is empty! Please define valid class names.")

NUM_CLASSES = len(CLASS_NAMES)

# ✅ ADD: Validate NUM_CLASSES > 0
if NUM_CLASSES == 0:
    raise ValueError("NUM_CLASSES is 0! Check CLASS_NAMES definition.")

IMAGE_SIZE = 224


# Model configurations
MODEL_CONFIGS = {
    'CustomCNN (LeafNetCNN)': {
        'path': 'custom_cnn_best.pth',
        'architecture': 'Custom CNN with conv layers (32→64→128→256→256)',
        'input_size': '224x224',
        'classes': NUM_CLASSES,
        'target_layer': 'conv4'
    },
    'ResNet18': {
        'path': 'resnet18_best.pth',
        'architecture': 'ResNet18 with modified fc layer',
        'input_size': '224x224',
        'classes': NUM_CLASSES,
        'target_layer': 'layer4[-1]'
    },
    'EfficientNet-B0': {
        'path': 'efficientnet_best.pth',
        'architecture': 'EfficientNet-B0 with custom classifier',
        'input_size': '224x224',
        'classes': NUM_CLASSES,
        'target_layer': 'features[-1]'
    },
    'DenseNet121': {
        'path': 'densenet121_best.pth',
        'architecture': 'DenseNet121 with modified classifier',
        'input_size': '224x224',
        'classes': NUM_CLASSES,
        'target_layer': 'features[-1]'
    },
    'SqueezeNet1.1': {
        'path': 'squeezenet_best.pth',
        'architecture': 'SqueezeNet1.1 with modified Conv2d classifier',
        'input_size': '224x224',
        'classes': NUM_CLASSES,
        'target_layer': 'features[10]'
    }
}


@st.cache_resource
def load_model(model_name):
    """Load the selected model with proper error handling"""
    try:
        config = MODEL_CONFIGS[model_name]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
        if model_name == 'CustomCNN (LeafNetCNN)':
            model = LeafNetCNN(NUM_CLASSES)
           
        elif model_name == 'ResNet18':
            model = resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
           
        # elif model_name == 'EfficientNet-B0':
        #     model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        #     for param in model.parameters():
        #         param.requires_grad = False
        #     model.classifier = nn.Sequential(
        #         nn.Dropout(p=0.5),
        #         nn.Linear(1280, 512),
        #         nn.SiLU(),
        #         nn.BatchNorm1d(512),
        #         nn.Dropout(p=0.3),
        #         nn.Linear(512, NUM_CLASSES)
        #     )


        elif model_name == 'EfficientNet-B0':
            # ✅ MODIFIED: Use weights, and DO NOT break features
            model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            # Freeze backbone
            for param in model.parameters():
                param.requires_grad = False
            # ✅ Only replace classifier, keep features intact
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(1280, 512),
                nn.SiLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(p=0.3),
                nn.Linear(512, NUM_CLASSES)
            )

        elif model_name == 'DenseNet121':
            model = densenet121(pretrained=True)
            model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
           
        elif model_name == 'SqueezeNet1.1':
            model = squeezenet1_1(pretrained=True)
            model.classifier[1] = nn.Conv2d(512, NUM_CLASSES, kernel_size=1)
            model.num_classes = NUM_CLASSES
       
        model.to(device)
       
        # Load weights with better error handling
        if os.path.exists(config['path']):
            checkpoint = torch.load(config['path'], map_location=device)
           
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
        else:
            st.error(f"Model file not found: {config['path']}")
            return None, None
       
        model.eval()
       
        # # Test the model with a dummy input to ensure it works
        # test_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
        # with torch.no_grad():
        #     test_output = model(test_input)
        #     if test_output.shape[1] != NUM_CLASSES:
        #         st.error(f"Model output shape mismatch. Expected {NUM_CLASSES}, got {test_output.shape[1]}")
        #         return None, None
        
        # # ✅ Debug: Confirm model loaded
        # st.write(f"✅ Model '{model_name}' loaded successfully.")
       
        # return model, config

        # ✅ DEBUG: Check if features exists in EfficientNet
        if model_name == 'EfficientNet-B0':
            if not hasattr(model, 'features') or model.features is None:
                st.error("❌ EfficientNet-B0: 'features' attribute missing or None!")
                return None, None

        # Test model
        test_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
        with torch.no_grad():
            test_output = model(test_input)
            if test_output.shape[1] != NUM_CLASSES:
                st.error(f"Model output shape mismatch. Expected {NUM_CLASSES}, got {test_output.shape[1]}")
                return None, None
       
        return model, config
       
    except Exception as e:
        st.error(f"Error loading model {model_name}: {str(e)}")
        import traceback
        st.error(f"Full traceback: {traceback.format_exc()}")
        return None, None


# def get_target_layer(model, model_name):
#     """Get the target layer for CAM methods"""
#     try:
#         if model_name == 'CustomCNN (LeafNetCNN)':
#             return [model.conv4]
#         elif model_name == 'ResNet18':
#             return [model.layer4[-1]]
#         elif model_name == 'EfficientNet-B0':
#             return [model.features[-1]]
#         elif model_name == 'DenseNet121':
#             return [model.features[-1]]
#         elif model_name == 'SqueezeNet1.1':
#             return [model.features[10]]
#         else:
#             return [list(model.children())[-2]]
#     except Exception as e:
#         st.error(f"Error getting target layer for {model_name}: {str(e)}")
#         # Return a default layer
#         return [list(model.children())[-2]]


# ✅ MODIFIED: get_target_layer — added fallback for EfficientNet
def get_target_layer(model, model_name):
    """Get the target layer for CAM methods with validation"""
    try:
        if model_name == 'CustomCNN (LeafNetCNN)':
            layer = model.conv4
        elif model_name == 'ResNet18':
            layer = model.layer4[-1]
        elif model_name == 'EfficientNet-B0':
            # ✅ Ensure features exists
            if not hasattr(model, 'features'):
                st.error("EfficientNet-B0: 'features' not found!")
                return None
            layer = model.features[-1]
        elif model_name == 'DenseNet121':
            layer = model.features[-1]
        elif model_name == 'SqueezeNet1.1':
            layer = model.features[10]
        else:
            children = list(model.children())
            if len(children) >= 2:
                layer = children[-2]
            else:
                return None

        if layer is None:
            st.error(f"Target layer for {model_name} is None!")
            return None

        return [layer]

    except Exception as e:
        st.error(f"Error getting target layer for {model_name}: {str(e)}")
        # Fallback: find last Conv2d
        for module in reversed(list(model.modules())):
            if isinstance(module, nn.Conv2d):
                st.warning(f"Fallback: using Conv2d at: {module}")
                return [module]
        return None


# ✅ MODIFIED: predict_image — added shape safety
def predict_image(model, image_tensor):
    """Make prediction on image with better error handling"""
    try:
        device = next(model.parameters()).device
        image_tensor = image_tensor.to(device)
       
        with torch.no_grad():
            outputs = model(image_tensor)

            # ✅ Validate shape
            if outputs.dim() < 2:
                st.error("Model output has invalid dimensions.")
                return None, None

            if outputs.shape[1] != NUM_CLASSES:
                st.error(f"Output shape mismatch: expected {NUM_CLASSES}, got {outputs.shape[1]}")
                return None, None

            probabilities = F.softmax(outputs, dim=1)
       
        return outputs, probabilities
       
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None, None


# ✅ MODIFIED: generate_gradcam — added layer debug
def generate_gradcam(model, image_tensor, model_name, cam_type='GradCAM'):
    """Generate CAM visualizations with error handling"""
    try:
        device = next(model.parameters()).device
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            if outputs.shape[1] != NUM_CLASSES:
                st.error("Invalid model output shape in GradCAM.")
                return None, None
            predicted_class = outputs.argmax(dim=1).item()

        targets = [ClassifierOutputTarget(predicted_class)]

        target_layers = get_target_layer(model, model_name)
        if target_layers is None or len(target_layers) == 0:
            st.error(f"❌ No valid target layer for {model_name}")
            return None, None

        # ✅ Debug: Show layer type
        layer = target_layers[0]
        st.write(f"🔧 CAM Layer ({cam_type}): {layer.__class__.__name__}")

        if cam_type == 'GradCAM':
            cam = GradCAM(model=model, target_layers=target_layers)
        elif cam_type == 'GradCAM++':
            cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
        elif cam_type == 'EigenCAM':
            cam = EigenCAM(model=model, target_layers=target_layers)
        elif cam_type == 'AblationCAM':
            cam = AblationCAM(model=model, target_layers=target_layers)
        else:
            st.error(f"Unknown CAM type: {cam_type}")
            return None, None

        try:
            grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
            if grayscale_cam is None or grayscale_cam.size == 0:
                st.error(f"{cam_type}: CAM is empty.")
                return None, None
            grayscale_cam = grayscale_cam[0, :]
            return grayscale_cam, predicted_class
        except Exception as cam_e:
            st.error(f"{cam_type}: CAM failed: {str(cam_e)}")
            return None, None

    except Exception as e:
        st.error(f"Error generating {cam_type}: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None, None


def preprocess_image(image):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
   
    if image.mode != 'RGB':
        image = image.convert('RGB')
   
    tensor = transform(image).unsqueeze(0)
    return tensor


# def predict_image(model, image_tensor):
#     """Make prediction on image with better error handling"""
#     try:
#         device = next(model.parameters()).device
#         image_tensor = image_tensor.to(device)
       
#         with torch.no_grad():
#             outputs = model(image_tensor)
#             # Ensure outputs is the correct shape
#             if len(outputs.shape) != 2 or outputs.shape[1] != NUM_CLASSES:
#                 raise ValueError(f"Invalid output shape: {outputs.shape}")
           
#             probabilities = F.softmax(outputs, dim=1)
       
#         return outputs, probabilities
       
#     except Exception as e:
#         st.error(f"Error in prediction: {str(e)}")
#         return None, None


# # ✅ MODIFIED: predict_image with robust shape checking
# def predict_image(model, image_tensor):
#     """Make prediction on image with better error handling"""
#     try:
#         device = next(model.parameters()).device
#         image_tensor = image_tensor.to(device)
       
#         with torch.no_grad():
#             outputs = model(image_tensor)

#             # ✅ ADD: Validate output shape
#             if outputs.dim() < 2:
#                 st.error("Model output has invalid dimensions (less than 2D).")
#                 return None, None

#             if outputs.shape[1] != NUM_CLASSES:
#                 st.error(f"Model output shape mismatch: expected {NUM_CLASSES}, got {outputs.shape[1]}")
#                 return None, None

#             probabilities = F.softmax(outputs, dim=1)
       
#         return outputs, probabilities
       
#     except Exception as e:
#         st.error(f"Error in prediction: {str(e)}")
#         return None, None

# def generate_gradcam(model, image_tensor, model_name, cam_type='GradCAM'):
#     """Generate CAM visualizations with error handling"""
 
#     try:
#         device = next(model.parameters()).device
#         image_tensor = image_tensor.to(device)
       
#         # Get predicted class first
#         with torch.no_grad():
#             outputs = model(image_tensor)
#             predicted_class = outputs.argmax(dim=1).item()
       
#         targets = [ClassifierOutputTarget(predicted_class)]
       
#         # Regular CNN models
#         target_layers = get_target_layer(model, model_name)
       
#         # Select CAM method
#         if cam_type == 'GradCAM':
#             cam = GradCAM(model=model, target_layers=target_layers)
#         elif cam_type == 'GradCAM++':
#             cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
#         elif cam_type == 'EigenCAM':
#             cam = EigenCAM(model=model, target_layers=target_layers)
#         elif cam_type == 'AblationCAM':
#             cam = AblationCAM(model=model, target_layers=target_layers)
       
#         # Generate CAM
#         grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
#         grayscale_cam = grayscale_cam[0, :]
       
#         return grayscale_cam, predicted_class
       
#     except Exception as e:
#         st.error(f"Error generating {cam_type}: {str(e)}")
#         return None, None


def generate_lime_explanation(model, original_image, image_tensor):
    """Generate LIME explanation with comprehensive error handling"""
    device = next(model.parameters()).device
   
    def predict_fn(images):
        """Prediction function for LIME"""
        try:
            predictions = []
           
            for img in images:
                # Ensure proper format
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                elif img.dtype != np.uint8:
                    img = img.astype(np.uint8)
               
                # Convert to PIL and preprocess
                pil_img = Image.fromarray(img)
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
               
                # Get tensor
                tensor = preprocess_image(pil_img).to(device)
               
                # Predict
                with torch.no_grad():
                    output = model(tensor)
                    prob = F.softmax(output, dim=1)
                    predictions.append(prob.cpu().numpy()[0])
           
            return np.array(predictions)
           
        except Exception as e:
            st.error(f"Prediction error in LIME: {str(e)}")
            # Return uniform random predictions as fallback
            return np.ones((len(images), NUM_CLASSES)) / NUM_CLASSES
   
    try:
        # Import scikit-image
        from skimage.segmentation import mark_boundaries
       
        # Prepare image
        img_array = np.array(original_image.resize((IMAGE_SIZE, IMAGE_SIZE)))
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8)
       
        # Create LIME explainer
        explainer = lime_image.LimeImageExplainer()
       
        # Get predicted class first
        with torch.no_grad():
            outputs = model(image_tensor.to(device))
            predicted_class = outputs.argmax(dim=1).item()
       
        # Generate explanation
        explanation = explainer.explain_instance(
            img_array,
            predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=100,  # Reduced for faster processing
            random_seed=42
        )
       
        # Get explanation image and mask
        temp, mask = explanation.get_image_and_mask(
            predicted_class,
            positive_only=True,
            num_features=8,
            hide_rest=False
        )
       
        # Create boundary visualization
        lime_boundaries = mark_boundaries(
            temp / 255.0,
            mask,
            color=(0, 1, 0),
            mode='thick'
        )
       
        lime_result = (lime_boundaries * 255).astype(np.uint8)
        return lime_result, mask, predicted_class
       
    except ImportError:
        st.error("scikit-image not properly installed. Please install it with: pip install scikit-image")
        fallback_img = np.array(original_image.resize((IMAGE_SIZE, IMAGE_SIZE)))
        return fallback_img, np.zeros((IMAGE_SIZE, IMAGE_SIZE)), 0
       
    except Exception as e:
        st.error(f"Error generating LIME explanation: {str(e)}")
        fallback_img = np.array(original_image.resize((IMAGE_SIZE, IMAGE_SIZE)))
        return fallback_img, np.zeros((IMAGE_SIZE, IMAGE_SIZE)), 0


def create_overlay_image(original_image, cam, alpha=0.4):
    """Create overlay of CAM on original image"""
    try:
        # Resize original image
        original_resized = cv2.resize(np.array(original_image), (IMAGE_SIZE, IMAGE_SIZE))
        original_resized = original_resized / 255.0
       
        # Create heatmap
        visualization = show_cam_on_image(original_resized, cam, use_rgb=True)
        return visualization
    except Exception as e:
        st.error(f"Error creating overlay: {str(e)}")
        return np.array(original_image.resize((IMAGE_SIZE, IMAGE_SIZE)))


def plot_explanations(original_image, explanations):
    """Plot all explanations in a grid"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('XAI Explanations Comparison', fontsize=16)
   
    # Original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
   
    # Plot each explanation
    titles = ['Grad-CAM', 'Grad-CAM++', 'Eigen-CAM', 'Ablation-CAM', 'LIME']
    positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
   
    for i, (explanation, title) in enumerate(zip(explanations, titles)):
        row, col = positions[i]
        axes[row, col].imshow(explanation)
        axes[row, col].set_title(title)
        axes[row, col].axis('off')
   
    plt.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="TOM24 XAI App", layout="wide")
   
    st.title("🌱 TOM24 Crop Disease & Pest Classification with XAI")
    st.markdown("### Explainable AI for Agricultural Image Analysis")
   
    # Sidebar
    st.sidebar.header("Configuration")
   
    # Model selection
    model_name = st.sidebar.selectbox(
        "Select Model",
        list(MODEL_CONFIGS.keys())
    )
   
    # Display model metadata
    if model_name:
        config = MODEL_CONFIGS[model_name]
        st.sidebar.subheader("Model Information")
        st.sidebar.write(f"**Architecture:** {config['architecture']}")
        st.sidebar.write(f"**Input Size:** {config['input_size']}")
        st.sidebar.write(f"**Classes:** {config['classes']}")
        st.sidebar.write(f"**Checkpoint:** {config['path']}")
        st.sidebar.write(f"**Target Layer:** {config['target_layer']}")
   
    # Main content
    col1, col2 = st.columns([1, 2])
   
    with col1:
        st.subheader("Image Input")
       
        # Image upload
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=['jpg', 'jpeg', 'png']
        )
       
        st.write("*Upload your own crop/plant images for testing*")
       
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
           
            # Load model
            with st.spinner("Loading model..."):
                model, model_config = load_model(model_name)
           
            if model is not None:
                # Preprocess image
                image_tensor = preprocess_image(image)
               
                # Make prediction
                with st.spinner("Making prediction..."):
                    outputs, probabilities = predict_image(model, image_tensor)
               
                if outputs is not None and probabilities is not None:
                    # Display predictions
                    st.subheader("Predictions")
                    probs_np = probabilities.cpu().numpy()[0]
                    top3_indices = np.argsort(probs_np)[-3:][::-1]
                   
                    for i, idx in enumerate(top3_indices):
                        confidence = probs_np[idx] * 100
                        st.write(f"{i+1}. **{CLASS_NAMES[idx]}**: {confidence:.2f}%")
                   
                    predicted_class = outputs.argmax(dim=1).item()
                    st.success(f"**Final Prediction: {CLASS_NAMES[predicted_class]}**")
                   
                    # Store prediction results for XAI
                    st.session_state.prediction_ready = True
                    st.session_state.model = model
                    st.session_state.image = image
                    st.session_state.image_tensor = image_tensor
                    st.session_state.model_name = model_name
                else:
                    st.error("Prediction failed. Please check your model and try again.")
            else:
                st.error("Failed to load model. Please check the model files.")
   
    with col2:
        if uploaded_file is not None and hasattr(st.session_state, 'prediction_ready') and st.session_state.prediction_ready:
            st.subheader("XAI Explanations")
           
            # Generate explanations
            with st.spinner("Generating explanations..."):
                explanations = []
                explanation_names = []
               
                # CAM methods
                cam_methods = ['GradCAM', 'GradCAM++', 'EigenCAM', 'AblationCAM']
                for cam_type in cam_methods:
                    try:
                        result = generate_gradcam(st.session_state.model, st.session_state.image_tensor, st.session_state.model_name, cam_type)
                        if result[0] is not None:
                            cam, pred_class = result
                            overlay = create_overlay_image(st.session_state.image, cam)
                            explanations.append(overlay)
                            explanation_names.append(cam_type)
                        else:
                            st.warning(f"Could not generate {cam_type}")
                    except Exception as e:
                        st.warning(f"Could not generate {cam_type}: {str(e)}")
               
                # LIME
                st.info("Generating LIME explanation (this may take a moment)...")
                try:
                    lime_image_exp, lime_mask, lime_pred = generate_lime_explanation(
                        st.session_state.model, st.session_state.image, st.session_state.image_tensor)
                    explanations.append(lime_image_exp)
                    explanation_names.append('LIME')
                    st.success("LIME explanation generated successfully!")
                except Exception as e:
                    st.error(f"Could not generate LIME: {str(e)}")
           
            # Display explanations in grid
            if explanations:
                cols = st.columns(3)
                for i, (explanation, name) in enumerate(zip(explanations, explanation_names)):
                    with cols[i % 3]:
                        st.image(explanation, caption=name, use_container_width=True)
               
                # Create downloadable visualization
                if len(explanations) >= 4:
                    fig = plot_explanations(st.session_state.image.resize((IMAGE_SIZE, IMAGE_SIZE)), explanations)
                   
                    # Save plot to buffer
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                    buf.seek(0)
                   
                    st.download_button(
                        label="📥 Download All Explanations",
                        data=buf.getvalue(),
                        file_name=f"xai_explanations_{st.session_state.model_name.replace(' ', '_')}.png",
                        mime="image/png"
                    )
                   
                    plt.close(fig)
   
    # Class information
    st.sidebar.subheader("Dataset Classes")
    for i, class_name in enumerate(CLASS_NAMES):
        st.sidebar.write(f"{i}: {class_name}")


if __name__ == "__main__":
    main()
