import os
from dotenv import load_dotenv
from openai import OpenAI

# Load .env file and keys
load_dotenv()

# Create OpenAI client with API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Open and transcribe audio
with open("input.wav", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )

# Print the transcribed text
print("📝 Transcription:", transcript.text)
