import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from lime import lime_image
import shap
import os
from train import create_model, explain_prediction

# Constants
IMG_SIZE = (224, 224)

def load_model():
    """Load the trained model"""
    model = create_model()
    if os.path.exists('models/best_model.h5'):
        model.load_weights('models/best_model.h5')
    return model

def preprocess_image(image):
    """Preprocess the uploaded image"""
    image = image.resize(IMG_SIZE)
    img_array = np.array(image)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def plot_lime_explanation(explanation, image):
    """Plot LIME explanation"""
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False
    )
    plt.imshow(mask)
    plt.axis('off')
    return plt

def plot_shap_explanation(shap_values, image):
    """Plot SHAP explanation"""
    shap.image_plot(shap_values, image)
    return plt

def main():
    st.title("Plant Disease Detection with Explainable AI")
    st.write("Upload a plant leaf image to detect diseases and get explanations")
    
    # Load model
    model = load_model()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Preprocess image
        img_array = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        st.write(f"Predicted Disease: Class {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}")
        
        # Generate explanations
        st.write("### Model Explanations")
        
        # LIME explanation
        st.write("#### LIME Explanation")
        explanation, _ = explain_prediction(model, uploaded_file, None)
        lime_plot = plot_lime_explanation(explanation, img_array[0])
        st.pyplot(lime_plot)
        
        # SHAP explanation
        st.write("#### SHAP Explanation")
        _, shap_values = explain_prediction(model, uploaded_file, None)
        shap_plot = plot_shap_explanation(shap_values, img_array[0])
        st.pyplot(shap_plot)
        
        # Additional information
        st.write("### About the Explanations")
        st.write("""
        - **LIME (Local Interpretable Model-agnostic Explanations)**: Shows which parts of the image
          contributed most to the prediction by highlighting important regions.
        
        - **SHAP (SHapley Additive exPlanations)**: Provides a more detailed explanation by showing
          how each pixel contributes to the prediction, with red indicating positive contributions
          and blue indicating negative contributions.
        """)

if __name__ == "__main__":
    main() 