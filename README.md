# TOM2024: Agricultural Crop Pest & Disease Classifier with XAI
An Explainable AI (XAI) production-ready web application built using Streamlit and PyTorch. This application classifies crop pests and diseases across tomato, onion, and maize plants using five deep learning architectures. It leverages advanced interpretability frameworks (Grad-CAM and LIME) to visually explain how the models diagnose plant pathology.

The core models are trained on the public TOM2024 Dataset via Mendeley Data, designed for developing edge-ready classification models in precision agriculture.

## 📺 Video Demonstration


## 🌾 Core Features
**Dataset Backbone:** Built explicitly around the TOM2024 benchmark dataset (Tomato, Onion, Maize images for pests and disease classification).

**Multi-Model Evaluation:** Compare crop diagnostic performance across five distinct architectures: Custom CNN, ResNet-18, EfficientNet, DenseNet-121, and SqueezeNet.

**Visual Diagnostics (Grad-CAM):** Highlights specific physiological regions on leaves and crops (e.g., lesions, mold patches, pest spots) that drove the model's decision.

**Superpixel Perturbation (LIME):** Sections crop images into local visual features to determine which regions positively or negatively support the health classification.

**Agile Codebase:** Bulky model weights are hosted on Hugging Face Hub and downloaded dynamically during the initial local runtime, keeping git version control clean and lightweight.

## 📊 Dataset Reference
**Dataset Name:** TOM2024: Datasets of tomato, onion, and maize images for developing pests and diseases AI-based classification models
**Source:** Mendeley Data Repository (v1)
**Application Scope:** Computer Vision for Precision Agriculture / Crop Disease Diagnostics

## 💻 Local Setup & Installation
Follow the steps below to initialize your virtual environment, satisfy dependencies, and run the analytical dashboard locally.

### Option A: Windows Command Prompt (cmd)
Bash  
**1. Create the virtual environment**  
python -m venv xai_env

**2. Activate the environment**  
xai_env\Scripts\activate.bat 

**3. Upgrade package manager**  
pip install --upgrade pip

**4. Install explainability framework**  
pip install grad-cam

**5. Install comprehensive deep learning, vision, and UI requirements**  
pip install torch torchvision timm lime scikit-image opencv-python Pillow numpy matplotlib streamlit

**6. Pull remaining environment dependencies**  
pip install -r requirements.txt

**7. Boot the Streamlit application**  
streamlit run app.py

### Option B: Git Bash (Windows) / macOS / Linux
Bash  
**1. Create the virtual environment**  
py -m venv xai_env

**2. Activate the environment**  
source xai_env/Scripts/activate

**3. Upgrade package manager**  
py -m pip install --upgrade pip

**4. Install explainability framework**  
py -m pip install grad-cam

**5. Install comprehensive deep learning, vision, and UI requirements**  
py -m pip install torch torchvision timm lime scikit-image opencv-python Pillow numpy matplotlib streamlit

**6. Pull remaining environment dependencies**  
py -m pip install -r requirements.txt

**7. Boot the Streamlit application**  
py -m streamlit run app.py

## 🧠 Automated Weight Management
To optimize version control, model parameter weights (.pth binaries) are explicitly decoupled from GitHub tracking rules via .gitignore.

On application initialization, app.py queries our public Hugging Face Hub Repository to securely pull missing files directly to the root environment directory:
custom_cnn_best.pth
resnet18_best.pth
efficientnet_best.pth
densenet121_best.pth
squeezenet_best.pth

## 📄 License
This project is open-source and available under the terms of the MIT License.
