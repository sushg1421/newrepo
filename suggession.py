import random
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from googletrans import Translator
from gtts import gTTS
import pygame
import time
# Load your data
df = pd.read_csv("dataset - Sheet1.csv")  # Replace with your dataset path

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define medical keywords for fallback
medical_keywords = {
    "fever": "It sounds like you may have a fever. Stay hydrated and consider seeing a doctor if symptoms persist.",
    "cough": "A persistent cough might be due to an infection or allergy. Try warm fluids and rest.",
    "headache": "Headaches can have many causes, including stress and dehydration. Consider resting and drinking water.",
    "cold": "Common colds usually go away on their own. Stay warm, drink fluids, and get rest.",
}

# List of health tips categorized by keywords
health_tips = {
    "sleep": [
        "Try to get at least 7-8 hours of sleep each night.",
        "Establish a regular sleep routine to improve sleep quality.",
        "Avoid screens before bed to help your mind relax.",
    ],
    "energy": [
        "Make sure you're eating a balanced diet to maintain energy.",
        "Exercise regularly to boost your energy levels.",
        "Stay hydrated throughout the day to avoid fatigue.",
    ],
    "stress": [
        "Take short breaks throughout the day to reduce stress.",
        "Practice mindfulness or meditation to help manage stress.",
        "Engage in physical activity to reduce anxiety and stress.",
    ],
    "general": [
        "Drink plenty of water throughout the day.",
        "Get at least 30 minutes of exercise every day.",
        "Eat a balanced diet rich in fruits and vegetables.",
    ],
}


# Function to get personalized health tip
def get_personalized_health_tip(user_input):
    # Convert input to lowercase for easier matching
    user_input_lower = user_input.lower()

    # Check for specific keywords in the user input
    if "tired" in user_input_lower or "fatigue" in user_input_lower:
        return random.choice(health_tips["energy"])
    elif "sleep" in user_input_lower or "rest" in user_input_lower:
        return random.choice(health_tips["sleep"])
    elif "stress" in user_input_lower or "anxious" in user_input_lower:
        return random.choice(health_tips["stress"])
    else:
        # Default to a general health tip if no specific keywords match
        return random.choice(health_tips["general"])


# Function to find the best cure based on similarity
def find_best_cure(user_input):
    user_input_embedding = model.encode(user_input, convert_to_tensor=True)
    disease_embeddings = model.encode(df['disease'].tolist(), convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(user_input_embedding, disease_embeddings)[0]
    best_match_idx = similarities.argmax().item()
    best_match_score = similarities[best_match_idx].item()

    # Define a similarity threshold for valid matches
    SIMILARITY_THRESHOLD = 0.5  # Adjust as needed

    if best_match_score < SIMILARITY_THRESHOLD:
        # Check for keywords in user input
        for keyword, response in medical_keywords.items():
            if keyword in user_input.lower():
                return response

        # Default fallback response if no keywords match
        return "I'm sorry, I don't have enough information on this. Please consult a healthcare professional."

    return df.iloc[best_match_idx]['cure']


# Function to translate text
def translate_text(text, dest_language='en'):
    return translator.translate(text, dest=dest_language).text


# Initialize translator
translator = Translator()

# Streamlit UI
st.title("Medical Chatbot 🤖")
user_input = st.text_input("Ask a question:")

# Language selection (user chooses from the updated list of languages)
language_choice = st.selectbox("Select Language", [
    "English", "Hindi", "Gujarati", "Korean", "Turkish",
    "German", "French", "Arabic", "Urdu", "Tamil", "Telugu", "Chinese", "Japanese","Kannada"
])

# Language codes based on the user selection
language_codes = {
    "English": "en",
    "Hindi": "hi",
    "Gujarati": "gu",
    "Korean": "ko",
    "Turkish": "tr",
    "German": "de",
    "French": "fr",
    "Arabic": "ar",
    "Urdu": "ur",
    "Tamil": "ta",
    "Telugu": "te",
    "Chinese": "zh-CN",  # Simplified Chinese
    "Japanese": "ja",
    "Kannada":"kn",
}

# Button for response
'''if st.button("Get Response"):
    if user_input:
        response = find_best_cure(user_input)
        # Translate the response based on the selected language
        translated_response = translate_text(response, dest_language=language_codes[language_choice])
        st.write(f"**My Suggestion is:** {translated_response}")


        text = translated_response
        tts = gTTS(text=text)
        tts.save("test_kannada.mp3")

        # Play audio using pygame
        pygame.mixer.init()
        pygame.mixer.music.load("test_kannada.mp3")
        pygame.mixer.music.play()

        # Wait till the audio finishes
        while pygame.mixer.music.get_busy():
            time.sleep(1)
        st.write("*Please note, the translation is provided by AI and might not be perfect.*")'''
import streamlit as st
from gtts import gTTS
import io
import base64

def render_audio_controls(audio_bytes):
    b64 = base64.b64encode(audio_bytes).decode()
    html = f"""
    <audio id="audio-player" controls autoplay>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    <br>
    <button onclick="document.getElementById('audio-player').pause()">⏸️ Stop Audio</button>
    <button onclick="document.getElementById('audio-player').currentTime=0; document.getElementById('audio-player').play()">🔁 Play Again</button>
    """
    st.markdown(html, unsafe_allow_html=True)

if st.button("Get Response"):
    if user_input:
        response = find_best_cure(user_input)
        dest_lang = language_codes[language_choice]
        translated_response = translate_text(response, dest_language=dest_lang)

        st.write(f"**My Suggestion is:** {translated_response}")

        # Generate speech
        tts = gTTS(text=translated_response, lang=dest_lang)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        audio_bytes = mp3_fp.read()

        render_audio_controls(audio_bytes)

        st.write("*Please note, the translation is provided by AI and might not be perfect.*")



# Add a button to get a personalized health tip
if st.button("Get a Personalized Health Tip"):
    if user_input:
        personalized_tip = get_personalized_health_tip(user_input)
        translated_tip = translate_text(personalized_tip, dest_language=language_codes[language_choice])
        st.write(f"**Health Tip:** {translated_tip}")
        st.write("*Please note, the translation is provided by AI and might not be perfect.*")