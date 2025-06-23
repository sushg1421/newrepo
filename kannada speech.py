import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import os

# Step 1: Record audio
fs = 44100  # Sample rate
seconds = 6  # Duration
print("🎙️ ಕನ್ನಡದಲ್ಲಿ ಮಾತಾಡಿ...")
recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()
filename = "input.wav"
write(filename, fs, recording)
print("✅ Audio saved as", filename)

# Step 2: Load Whisper model
print("📦 Loading Whisper model (medium)...")
model = whisper.load_model("medium")  # Use "base" or "small" if slow

# Step 3: Transcribe with forced Kannada language
print("🔍 Transcribing...")
result = model.transcribe(filename, language="kn", task="transcribe")



# Step 4: Print results
print("🌍 Detected Language:", result.get("language", "unknown"))
print("📝 Kannada Transcription:", result["text"])