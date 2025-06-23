from googletrans import Translator

translator = Translator()

# Sample local language sentence
text = input("enter text:")

# Translate to English
result = translator.translate(text, src='kn', dest='en')
print("🗣️ Translated Text:", result.text)
