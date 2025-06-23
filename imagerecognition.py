import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the MobileNetV2 model pre-trained on ImageNet
model = MobileNetV2(weights='imagenet')

# Load your image
img_path = 'bike.jpg'  # change to your image file
img = image.load_img(img_path, target_size=(224, 224))

# Convert image to array and preprocess
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Predict
preds = model.predict(x)

# Decode and display the top prediction
decoded = decode_predictions(preds, top=3)[0]
for i, (imagenet_id, label, confidence) in enumerate(decoded):
    print(f"{i+1}. {label} ({confidence * 100:.2f}%)")
