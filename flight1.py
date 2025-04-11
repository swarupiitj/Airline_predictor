import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from PIL import ImageOps
import matplotlib.pyplot as plt
import os

# --- Load model and set device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the CNN model (same as your training script)
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Set your classes manually (since ImageFolder isn't used here)
class_names = ['Air India','Emirates','Etihad','Indigo', 'Qatar']  

model = SimpleCNN(num_classes=len(class_names))
model.load_state_dict(torch.load("simple_airline_model.pth", map_location=device))
model.to(device)
model.eval()

# --- Define transformation ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
# --- Streamlit UI ---
st.set_page_config(page_title="✈️ Airline Recogniser", layout="centered")
#st.title("🛫 Airline Recogniser")
st.write("<h1 style='text-align: center; color: blue;'>🛫 Airline Recogniser 🛫</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: yellow;'>Upload an airline image (EMIRATES, ETIHAD, QATAR, INDIGO, AIR INDIA), and the model will predict the airline!</h4>", unsafe_allow_html=True)


uploaded_file = st.file_uploader("Upload your Image below", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    square_image = ImageOps.fit(image, (300, 300), method=Image.Resampling.LANCZOS)
    st.image(square_image, caption=None, use_container_width=False)

    # Preprocess and predict
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        prediction = class_names[predicted.item()]

    st.success(f"✅ The above picture is related to: **{prediction}** Airline")
