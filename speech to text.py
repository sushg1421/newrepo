import whisper
import sounddevice as sd
from scipy.io.wavfile import write
from googletrans import Translator

translator = Translator()
# Step 1: Record audio
fs = 44100
seconds = 5
print("🎙️ Speak now...")
recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()
filename = "input.wav"
write(filename, fs, recording)
print("✅ Audio saved as", filename)

# Step 2: Transcribe using Whisper locally
model = whisper.load_model("base")  # or "small", "medium", "large"
result = model.transcribe(filename)
print("📝 Transcription:", result["text"])
data = translator.translate(result["text"], src='kn', dest='en')
print("🗣️ Translated Text:", data.text)


import whisper
import pyttsx3

# Load and transcribe
model = whisper.load_model("base")
filename = "input.wav"
result = model.transcribe(filename)
transcribed_text = data["text"]

print("📝 Transcription:", transcribed_text)

# Convert to speech
engine = pyttsx3.init()
engine.say(transcribed_text)
engine.runAndWait()
