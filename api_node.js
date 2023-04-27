const express = require('express'); // Importa o framework Express
const tf = require('@tensorflow/tfjs-node'); // Importa a biblioteca TensorFlow.js para Node.js
const fs = require('fs'); // Importa o módulo fs para trabalhar com arquivos
const sharp = require('sharp'); // Importa a biblioteca Sharp para redimensionar imagens
const multer = require('multer'); // Importa a biblioteca Multer para lidar com o upload de arquivos

const app = express(); // Cria uma instância do Express
const upload = multer({ dest: 'uploads/' }); // Cria uma instância do Multer para lidar com o upload de arquivos

class ImageClassifier {
  constructor(modelPath) { // Construtor da classe
    this.model = tf.node.loadSavedModel(modelPath); // Carrega o modelo TensorFlow.js do caminho especificado
  }

  async preprocessImage(imagePath) { // Função para pré-processar a imagem
    const buffer = await sharp(imagePath).resize(224, 224).toBuffer(); // Redimensiona a imagem e a converte para um buffer
    const image = tf.node.decodeImage(buffer, 3); // Converte o buffer em um tensor
    const finalImage = image.expandDims(0).toFloat().div(255.0); // Normaliza os valores dos pixels para o intervalo [0, 1]
    return finalImage; // Retorna a imagem pré-processada
  }

  async predict(image) { // Função para fazer a predição da imagem
    const prediction = await this.model.predict(image); // Executa a inferência na imagem de entrada
    const classIdx = tf.argMax(prediction, axis=-1).dataSync()[0]; // Obtém o índice da classe com maior probabilidade
    const classNames = ['benign', 'normal', 'malignant']; // Nomes das classes
    const classLabel = classNames[classIdx]; // Define a classe correspondente ao índice
    const probability = prediction.dataSync()[classIdx]; // Obtém a probabilidade da classe com maior probabilidade
    return { class: classLabel, probability }; // Retorna a classe e a probabilidade da classe com maior probabilidade
  }
}

const classifier = new ImageClassifier('BreastCancerSegmentor'); // Cria uma instância da classe ImageClassifier com o modelo especificado

app.post('/classify', upload.single('file'), async (req, res) => { // Define um endpoint /classify que só aceita solicitações POST com um único arquivo
  if (!req.file) { // Verifica se há um arquivo na solicitação
    return res.status(400).json({ error: 'No file uploaded' }); // Retorna um JSON com uma mensagem de erro e o status HTTP correspondente
  }
  try {
    const image = await classifier.preprocessImage(req.file.path); // Pré-processa a imagem usando a instância de ImageClassifier
    const result = await classifier.predict(image); // Executa a classificação usando a instância de ImageClassifier
    if (result.probability < 0.9) { // Verifica se a probabilidade da classe com maior probabilidade é menor que 0.9
      return res.status(400).json({ error: 'Unable to classify image with high confidence.' }); // Retorna um JSON com uma mensagem de erro e o status HTTP correspondente
    }
    return res.json(result); // Retorna um JSON com a classe e a probabilidade da classe com maior probabilidade e o status HTTP correspondente
  } catch (err} catch (err) {
return res.status(500).json({ error: 'Internal server error' }); // Retorna um JSON com uma mensagem de erro e o status HTTP correspondente caso ocorra um erro interno do servidor
}
});

app.listen(3000, () => { // Inicia o servidor na porta 3000
console.log('Server listening on port 3000'); // Exibe uma mensagem no console informando que o servidor está ouvindo na porta especificada
});
