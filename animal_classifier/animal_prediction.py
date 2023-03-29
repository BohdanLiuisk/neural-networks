import os
import sys
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))

def predict_image(image_path):
    # Load the saved model
    model = load_model(os.path.join(script_dir, 'animal_classifier.h5'))

    # Open and preprocess the image
    img = Image.open(image_path)
    img_resized = img.resize((150, 150))
    img_array = img_to_array(img_resized)
    img_normalized = img_array / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)

    # Make predictions
    predictions = model.predict(img_expanded)[0]
    labels = ['cat', 'chicken', 'cow']

    # Find the label with the highest score
    max_score_index = np.argmax(predictions)
    predicted_label = labels[max_score_index]
    
    # Display the image and the predicted label
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_label}")
    plt.axis("off")
    plt.show()

    # Print the results
    for i in range(len(labels)):
        print(f"{labels[i]} (score = {predictions[i]:.2f})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
    else:
        predict_image(sys.argv[1])