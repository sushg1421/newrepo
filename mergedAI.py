import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from googletrans import Translator
from gtts import gTTS
import io
import base64
import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import os
from PIL import Image
from torchvision import transforms, models
from medmnist import INFO, ChestMNIST
import torchvision
import time

# -------------------- MODEL SETUP --------------------

# Load CSV data
csv_data = pd.read_csv("dataset - Sheet1.csv")

# SentenceTransformer
text_model = SentenceTransformer('all-MiniLM-L6-v2')

# Google Translate
translator = Translator()

# Whisper (speech-to-text)
whisper_model = whisper.load_model("base")

# ChestMNIST Setup
info = INFO['chestmnist']
num_classes = len(info['label'])

xray_model = models.resnet18(weights=None)
xray_model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
xray_model.fc = torch.nn.Sequential(
    torch.nn.Linear(xray_model.fc.in_features, num_classes),
    torch.nn.Sigmoid()
)
xray_model.load_state_dict(torch.load("chestmnist_pretrained.pth", map_location=torch.device('cpu')))
xray_model.eval()

# Disease labels and explanations
xray_labels = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
]

disease_explanations = {
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

# -------------------- FUNCTIONS --------------------

def find_best_cure(user_input):
    user_embedding = text_model.encode(user_input, convert_to_tensor=True)
    disease_embeddings = text_model.encode(csv_data['disease'].tolist(), convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(user_embedding, disease_embeddings)[0]
    best_match_idx = similarities.argmax().item()
    return csv_data.iloc[best_match_idx]['cure']

def translate_text(text, dest_language='en'):
    return translator.translate(text, dest=dest_language).text

def classify_xray(image):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = xray_model(img_tensor)
        probs = torch.sigmoid(output)[0]

    results = []
    for i, prob in enumerate(probs):
        if prob > 0.5:
            label = xray_labels[i]
            explanation = disease_explanations.get(label, "")
            results.append(f"{label} ({prob:.2f}) — {explanation}")
    return results if results else ["❌ No strong disease indication found."]

def render_audio_controls(audio_bytes):
    b64 = base64.b64encode(audio_bytes).decode()
    html = f"""
    <audio id=\"audio-player\" controls autoplay>
        <source src=\"data:audio/mp3;base64,{b64}\" type=\"audio/mp3\">
    </audio>
    <br>
    <button onclick=\"document.getElementById('audio-player').pause()\">⏸️ Stop Audio</button>
    <button onclick=\"document.getElementById('audio-player').currentTime=0; document.getElementById('audio-player').play()\">🔁 Play Again</button>
    """
    st.markdown(html, unsafe_allow_html=True)

# -------------------- STREAMLIT UI --------------------

st.title("🧠 AI Medical Assistant with Voice + X-ray Diagnosis")

# Voice-to-text input
if "spoken_text" not in st.session_state:
    st.session_state["spoken_text"] = ""

if st.button("🎙️ Speak Symptoms"):
    fs = 44100
    seconds = 5
    st.info("🎙️ Recording... Please speak")
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()
    filename = "input.wav"
    write(filename, fs, recording)
    st.success("✅ Recording complete")
    try:
        result = whisper_model.transcribe(filename)
        st.session_state["spoken_text"] = result["text"]
        st.success("📝 Transcription complete!")
    except:
        st.session_state["spoken_text"] = ""
    os.remove(filename)

# Language selector
language_choice = st.selectbox("🌍 Select Language", [
    "English", "Hindi", "Gujarati", "Korean", "Turkish",
    "German", "French", "Arabic", "Urdu", "Tamil", "Telugu", "Chinese", "Japanese", "Kannada"
])
language_codes = {
    "English": "en", "Hindi": "hi", "Gujarati": "gu", "Korean": "ko", "Turkish": "tr",
    "German": "de", "French": "fr", "Arabic": "ar", "Urdu": "ur", "Tamil": "ta",
    "Telugu": "te", "Chinese": "zh-CN", "Japanese": "ja", "Kannada": "kn"
}

# Text input
user_input = st.text_input("💬 Describe your symptoms:", st.session_state["spoken_text"])

if st.button("Get Cure Suggestion"):
    if user_input:
        response = find_best_cure(user_input)
        translated = translate_text(response, dest_language=language_codes[language_choice])
        st.write(f"**🩺 Suggested Cure:** {translated}")
        tts = gTTS(text=translated, lang=language_codes[language_choice])
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        render_audio_controls(mp3_fp.read())

# -------------------- X-ray Diagnosis --------------------
st.header("📷 Upload Chest X-ray for Diagnosis")
uploaded_image = st.file_uploader("Upload a chest X-ray", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    if st.button("Diagnose X-ray"):
        results = classify_xray(image)
        st.info("🧠 Predicted Conditions from X-ray:")
        for r in results:
            st.write(f"🔹 {r}")
