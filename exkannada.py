from gtts import gTTS
import pygame
import time

text = "ನಾನು ಬೆಂಗಳೂರಿಗೆ ನಿನ್ನೆ ಬಂದಿದ್ದೇನೆ"
tts = gTTS(text=text, lang='kn')
tts.save("test_kannada.mp3")

# Play audio using pygame
pygame.mixer.init()
pygame.mixer.music.load("test_kannada.mp3")
pygame.mixer.music.play()

# Wait till the audio finishes
while pygame.mixer.music.get_busy():
    time.sleep(1)
