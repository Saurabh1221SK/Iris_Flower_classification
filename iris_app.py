import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from sklearn import datasets
from PIL import Image

model = tf.keras.models.load_model("iris_model.keras")

iris = datasets.load_iris()

setosa_image_path = "setosa.jpg"
versicolor_image_path = "versicolor.jpg"
virginica_image_path = "virginica.jpg"

def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = np.argmax(model.predict(input_data), axis=-1)
    species = iris.target_names[prediction][0]
    return species

def load_image(image_path):
    img = Image.open(image_path)
    return img

st.title("Iris Flower Species Prediction App")

st.write("""
This app predicts the iris flower species based on the input parameters using a neural network.
""")

sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width", 0.1, 2.5, 1.0)

species_prediction = predict_species(sepal_length, sepal_width, petal_length, petal_width)

if species_prediction == "setosa":
    image = load_image(setosa_image_path)
elif species_prediction == "versicolor":
    image = load_image(versicolor_image_path)
else:
    image = load_image(virginica_image_path)

col1, col2 = st.columns([2, 1])
with col1:
    st.write(f"Neural Network Prediction: {species_prediction}")
with col2:
    st.image(image, caption=f"Image of {species_prediction} species", use_column_width=True)