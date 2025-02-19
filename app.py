import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Function to load the model
def load_model(model_path):
    model = models.densenet121(weights=None)  # Load DenseNet121 without pretrained weights
    num_features = model.classifier.in_features

    # Updated classifier to match the saved model structure
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),  # Added dropout layer for regularization (if used during training)
        nn.Linear(512, 3)  # Ensure this matches your number of classes
    )

    # Load the saved state dict
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # Remove potential mismatches in keys (e.g., if 'module.' prefix was added during saving)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model

# Image preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Prediction function
def predict(model, image):
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        classes = ['Benign', 'Malignant', 'Normal']  # Update if your classes differ
        return classes[predicted.item()]

# Streamlit app layout
st.title("Liver Tumor Classification App")
model_path = 'liver_tumor_densenet121.pth'

# Load the model
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")

uploaded_file = st.file_uploader("Choose a histopathology image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)

    if st.button("Predict"):
        preprocessed_image = preprocess_image(image)
        prediction = predict(model, preprocessed_image)
        st.success(f"Prediction: {prediction}")
