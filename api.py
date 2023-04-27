import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import cv2
from flask import Flask, request, jsonify

app = Flask(__name__) # Cria uma instância do Flask com o nome do módulo atual (__name__)

class ImageClassifier:
    def __init__(self, model_path):
        self.model = load_model(model_path) # Carrega o modelo Keras do caminho especificado

    def preprocess_image(self, image_path):
        original_image = Image.open(image_path) # Abre a imagem com PIL
        resized_image = original_image.resize((224, 224)) # Redimensiona a imagem para 224x224 pixels
        final_image = np.array(resized_image) / 255.0 # Normaliza os valores dos pixels para o intervalo [0, 1]
        final_image = np.expand_dims(final_image, axis=0) # Adiciona uma dimensão para ter formato (1, 224, 224, 3)
        return final_image
    
    def predict(self, image):
        prediction = self.model.predict(image).flatten() # Executa a inferência na imagem de entrada
        class_idx = np.argmax(prediction) # Obtém o índice da classe com maior probabilidade
        class_label = "benign" if class_idx == 0 else "normal" if class_idx == 1 else "malignant" # Define a classe correspondente ao índice
        return class_label, prediction[class_idx] # Retorna a classe e a probabilidade da classe com maior probabilidade

classifier = ImageClassifier('BreastCancerSegmentor.h5') # Cria uma instância da classe ImageClassifier com o modelo especificado

@app.route('/classify', methods=['POST']) # Define um endpoint /classify que só aceita solicitações POST
def classify():
    if 'file' not in request.files: # Verifica se há um arquivo com o nome 'file' na solicitação
        return jsonify({'error': 'No file uploaded'}) # Retorna um JSON com uma mensagem de erro
    
    image_file = request.files['file'] # Obtém o arquivo de imagem da solicitação
    image = classifier.preprocess_image(image_file) # Pré-processa a imagem usando a instância de ImageClassifier
    class_label, probability = classifier.predict(image) # Executa a classificação usando a instância de ImageClassifier

    if probability < 0.9: # Verifica se a probabilidade da classe com maior probabilidade é menor que 0.9
        return jsonify({'error': 'Unable to classify image with high confidence.'}) # Retorna um JSON com uma mensagem de erro
    
    return jsonify({'class': class_label, 'probability': float(probability)}) # Retorna um JSON com a classe e a probabilidade da classe com maior probabilidade

if __name__ == '__main__':
    app.run() # Inicia o servidor web Flask para atender as solicitações dos clientes
