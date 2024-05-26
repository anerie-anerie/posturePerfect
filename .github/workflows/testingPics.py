import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model("model2.keras")

# Define a function to predict on a single image
def predict_image(image_path, model):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    return prediction[0][0]

# Path to the goodPos and badPos directories
good_pos_dir = "goodTests"
bad_pos_dir = "badTests"

# Initialize lists to store predictions and labels
good_pos_predictions = []
bad_pos_predictions = []

# Predictions for goodPos images
for img_file in os.listdir(good_pos_dir):
    img_path = os.path.join(good_pos_dir, img_file)
    prediction = predict_image(img_path, model)
    good_pos_predictions.append(prediction)

# Predictions for badPos images
for img_file in os.listdir(bad_pos_dir):
    img_path = os.path.join(bad_pos_dir, img_file)
    prediction = predict_image(img_path, model)
    bad_pos_predictions.append(prediction)

# Plotting
plt.figure(figsize=(10, 5))
print(good_pos_predictions)
print(bad_pos_predictions)