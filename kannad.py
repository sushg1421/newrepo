import whisper
import sounddevice as sd
from scipy.io.wavfile import write
from googletrans import Translator
import pyttsx3

# Step 1: Record audio
fs = 44100
seconds = 6
print("🎙️ Speak now...")
recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()
filename = "input.wav"
write(filename, fs, recording)
print("✅ Audio saved as", filename)
# Step 2: Transcribe using Whisper
model = whisper.load_model("base")
transcription_result = model.transcribe(filename)
original_text = transcription_result["text"]
print("📝 Transcription:", original_text)

# Step 3: Translate to Kannada
translator = Translator()
translated_result = translator.translate(original_text, src="en", dest='kn')
translated_text = translated_result.text
print("🌐 Translated Text (Kannada):", translated_text)

# Step 4: Speak the translated text
'''engine = pyttsx3.init()

# Set voice to Indian (optional, if supported)
voices = engine.getProperty('voices')
for voice in voices:
    if 'kannada' in voice.name.lower() or 'india' in voice.name.lower():
        engine.setProperty('voice', voice.id)
        break

engine.say(translated_text)
engine.runAndWait()'''
from gtts import gTTS
import pygame
import time

text = translated_text
tts = gTTS(text=text, lang='kn')
tts.save("test_kannada.mp3")

# Play audio using pygame
pygame.mixer.init()
pygame.mixer.music.load("test_kannada.mp3")
pygame.mixer.music.play()

# Wait till the audio finishes
while pygame.mixer.music.get_busy():
    time.sleep(1)
