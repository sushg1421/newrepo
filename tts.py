from dotenv import load_dotenv
import os
import requests
import boto3
from botocore.exceptions import NoCredentialsError
from botocore.config import Config
#from conversation_logic import generate_openai_response

load_dotenv()

# AWS S3 bucket details
S3_BUCKET_NAME = "jhubuckethophacks"
S3_OBJECT_NAME = "doctor1.mp3"  # Name of the file when uploaded to S3
S3_REGION = "us-east-2"  # Ensure this matches your bucket's region

# Language voice mapping
language_voice_map = {
    "en": {"voice_id": "mCQMfsqGDT6IDkEKR20a", "language": "English"},
    "hi": {"voice_id": "mCQMfsqGDT6IDkEKR20a", "language": "Hindi"},
    "ta": {"voice_id": "mCQMfsqGDT6IDkEKR20a", "language": "Tamil"},
}

# Ensure correct signature version and region are used
my_config = Config(region_name="us-east-2", signature_version="s3v4")

# AWS credentials are assumed to be configured via environment or AWS CLI
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    config=my_config,
)


def translate_text(text, target_language):
    """
    Translates the given text to the target language using OpenAI's GPT model.
    """
    prompt = (
        f"Translate the following text to {target_language}:\n\n{text}\n\nTranslation:"
    )
    translated_text = generate_openai_response(prompt)
    return translated_text.strip()


def text_to_speech(text, language="en"):
    print(f"Starting text_to_speech function with text: {text[:50]}... and language: {language}")

    if language != "en":
        print(f"Translating text to {language}")
        text = translate_text(text, language_voice_map.get(language)["language"])
        print(f"Translated text: {text[:50]}...")

    CHUNK_SIZE = 1024
    voice_info = language_voice_map.get(language)
    voice_id = voice_info["voice_id"] if voice_info else None
    url = "https://api.elevenlabs.io/v1/text-to-speech/" + voice_id
    print(f"Using voice ID: {voice_id}")

    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": os.getenv("ELEVEN_API_KEY"),
    }
    print(f"API Key: {os.getenv('ELEVEN_API_KEY')[:5]}...")  # Print first 5 chars of API key

    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {"stability": 0.7, "similarity_boost": 0.8},
    }

    print("Sending request to Eleven Labs API")
    response = requests.post(url, json=data, headers=headers)
    print(f"Response status code: {response.status_code}")
    print(f"Response headers: {response.headers}")

    if response.status_code != 200 or response.headers.get("Content-Type") != "audio/mpeg":
        print("Failed to retrieve valid audio data.")
        print(f"Response text: {response.text}")
        return None

    print("Successfully received audio data")
    binary_audio_data = b""
    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
        if chunk:
            binary_audio_data += chunk
    print(f"Total audio data size: {len(binary_audio_data)} bytes")

    try:
        print(f"Uploading to S3 bucket: {S3_BUCKET_NAME}")
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=S3_OBJECT_NAME,
            Body=binary_audio_data,
            ContentType="audio/mpeg",
        )
        print(f"File uploaded successfully to https://{S3_BUCKET_NAME}.s3.{S3_REGION}.amazonaws.com/{S3_OBJECT_NAME}")

        print("Generating pre-signed URL")
        presigned_url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET_NAME, "Key": S3_OBJECT_NAME},
            ExpiresIn=3600,
        )
        print(f"Pre-signed URL generated: {presigned_url[:50]}...")
        return presigned_url
    except NoCredentialsError:
        print("Credentials not available. Please check your AWS credentials.")
    except Exception as e:
        print(f"Failed to upload file: {e}")

    print("Exiting text_to_speech function")