import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import cv2

class ImageClassifier:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def preprocess_image(self, image_path):
        original_image = Image.open(image_path)
        resized_image = original_image.resize((224, 224))
        final_image = np.array(resized_image) / 255.0
        final_image = np.expand_dims(final_image, axis=0)
        return final_image
    
    def predict(self, image):
        prediction = self.model.predict(image).flatten()
        class_idx = np.argmax(prediction)
        class_label = "benign" if class_idx == 0 else "normal" if class_idx == 1 else "malignant"
        return class_label, prediction[class_idx]
    
def ui():
    st.set_page_config(page_title="Breast Cancer Classification", page_icon=":microscope:", layout="wide")
    st.title("Breast Cancer Classification")
    st.subheader("A startup for classifying breast cancer tumors")

    model_path = st.selectbox("Select a model", ['BreastCancerSegmentor.h5'])
    classifier = ImageClassifier(model_path)

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = classifier.preprocess_image(uploaded_file)
        class_label, probability = classifier.predict(image)
        
        if probability < 0.9:
            st.write("Unable to classify image with high confidence.")
            return
        
        st.write("The image is classified as:", class_label)
        st.write(f"Probability: {format(probability, '.2f')}")

        binary_image = image.copy()
        binary_image[binary_image >= 0.5] = 1
        binary_image[binary_image < 0.5] = 0

        binary_image_pil = Image.fromarray(np.uint8(binary_image[0] * 255))
        binary_image_pil = cv2.resize(np.uint8(binary_image_pil), None, fx=1.2, fy=1.2)
        binary_image_pil = np.array(Image.fromarray(np.uint8(binary_image_pil)))
            
        # Apply the morphological operation of opening to remove small pixels and fill internal gaps
        kernel = np.ones((5,5),np.uint8)
        opening = cv2.morphologyEx(binary_image_pil, cv2.MORPH_OPEN, kernel)
        opening_pil = Image.fromarray(opening)
        opening_pil = opening_pil.resize((int(224 * 1.2), int(224 * 1.2)))
                                  
        col1, col2,col3 = st.columns(3)
        with col1:
            st.image(image[0], width=224)
        with col2:
            st.image(binary_image_pil, width=224)
        with col3:
            st.image(opening_pil, width=224)

if __name__ == '__main__':
    ui()
