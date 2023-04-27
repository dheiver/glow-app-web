const express = require('express');
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const sharp = require('sharp');
const multer = require('multer');

const app = express();
const upload = multer({ dest: 'uploads/' });

class ImageClassifier {
  constructor(modelPath) {
    this.model = tf.node.loadSavedModel(modelPath);
  }

  async preprocessImage(imagePath) {
    const buffer = await sharp(imagePath).resize(224, 224).toBuffer();
    const image = tf.node.decodeImage(buffer, 3);
    const finalImage = image.expandDims(0).toFloat().div(255.0);
    return finalImage;
  }

  async predict(image) {
    const prediction = await this.model.predict(image);
    const classIdx = tf.argMax(prediction, axis=-1).dataSync()[0];
    const classNames = ['benign', 'normal', 'malignant'];
    const classLabel = classNames[classIdx];
    const probability = prediction.dataSync()[classIdx];
    return { class: classLabel, probability };
  }
}

const classifier = new ImageClassifier('BreastCancerSegmentor');

app.post('/classify', upload.single('file'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }
  try {
    const image = await classifier.preprocessImage(req.file.path);
    const result = await classifier.predict(image);
    if (result.probability < 0.9) {
      return res.status(400).json({ error: 'Unable to classify image with high confidence.' });
    }
    return res.json(result);
  } catch (err) {
    return res.status(500).json({ error: 'Internal server error' });
  }
});

app.listen(3000, () => {
  console.log('Server listening on port 3000');
});

// Exemplo de como chamar a API usando o método fetch em uma aplicação React
fetch('/classify', {
  method: 'POST',
  body: formData // objeto FormData contendo a imagem a ser classificada
})
  .then(res => res.json())
  .then(data => {
    console.log(data);
    // fazer algo com os dados retornados pela API, como exibir a classe e a probabilidade da classe com maior probabilidade
  })
  .catch(error => {
    console.error(error);
    // tratar erros de rede ou erros retornados pela API
  });
