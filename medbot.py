import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from medmnist import INFO, ChestMNIST
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn

# ---------------- SETUP ----------------
# Load CSV disease dataset (for text-based lookup)
df = pd.read_csv("dataset - Sheet1.csv")  # Your disease-to-cure file

# SentenceTransformer model for text input
text_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load MedMNIST model (ChestMNIST example)
data_flag = 'chestmnist'
info = INFO[data_flag]
num_classes = len(info['label'])

# Load pretrained ChestMNIST model
model = models.resnet18(weights=None)
model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 14),
    nn.Sigmoid()
)


# ---------------- FUNCTIONS ----------------
def find_best_cure(user_input):
    user_vec = text_model.encode(user_input, convert_to_tensor=True)
    disease_vecs = text_model.encode(df["disease"].tolist(), convert_to_tensor=True)
    sims = util.pytorch_cos_sim(user_vec, disease_vecs)[0]
    idx = sims.argmax().item()
    return df.iloc[idx]["cure"]

def classify_xray(image):
    import torchvision.transforms as transforms
    from PIL import Image
    import torch

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    img = transform(image).unsqueeze(0)  # Add batch dim

    # Predict
    with torch.no_grad():
        output = model(img)
        probs = torch.sigmoid(output)[0]

    labels = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
        "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
        "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
    ]

    threshold = 0.5
    result = []

    for i, prob in enumerate(probs):
        if prob > threshold:
            result.append(f"{labels[i]} ({prob:.2f})")

    return result if result else ["❌ No strong disease indication found."]


# ---------------- STREAMLIT UI ----------------
st.title("🩺 AI Medical Assistant")

# TEXT INPUT
user_input = st.text_input("Describe your symptoms (e.g., I have a cough and fever):")
if st.button("Get Cure Suggestion"):
    if user_input:
        cure = find_best_cure(user_input)
        st.success(f"💊 Suggested Cure:\n{cure}")

# IMAGE UPLOAD
uploaded_image = st.file_uploader("Or upload an X-ray image:", type=["png", "jpg", "jpeg"])
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    if st.button("Diagnose X-ray"):
        results = classify_xray(image)

        explanations = {
            "Atelectasis": "Partial or complete collapse of the lung.",
            "Cardiomegaly": "Enlarged heart, often due to high blood pressure.",
            "Effusion": "Fluid around the lungs (pleural space).",
            "Infiltration": "Substances like fluid or cells in the lung tissue.",
            "Mass": "A large abnormal tissue growth.",
            "Nodule": "Small lump or growth in the lungs.",
            "Pneumonia": "Lung infection causing inflammation.",
            "Pneumothorax": "Collapsed lung due to air leakage.",
            "Consolidation": "Lung tissue filled with liquid instead of air.",
            "Edema": "Fluid buildup in the lungs (common in heart failure).",
            "Emphysema": "Air sacs in the lungs are damaged.",
            "Fibrosis": "Scarring of lung tissue, often from chronic illness.",
            "Pleural_Thickening": "Thickening of the lining of the lungs.",
            "Hernia": "Tissue pushes through chest wall or diaphragm."
        }

        st.info("🧠 Predicted Conditions from X-ray:")
        for label in results:
            name = label.split('(')[0].strip()  # Extract label name before probability
            st.write(f"🔹 {label} — {explanations.get(name, 'No info available.')}")

