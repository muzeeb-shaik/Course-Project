import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


import numpy as np

app = Flask(__name__)

# Paths for Saved Models and Test X-Ray Scan
MODEL_PATH = 'models/saved_model.h5'
IMAGE_PATH = 'uploads'

# Loading previously trained and saved model
model = load_model(MODEL_PATH)
print('Model loaded. Check http://127.0.0.1:5000/')

def model_predict(img, model):

    datagen = ImageDataGenerator(rescale=1. / 255)
    x = datagen.flow_from_directory(IMAGE_PATH,target_size=(64, 64), color_mode='grayscale', batch_size=1, class_mode=None,shuffle=False)
    preds = model.predict(x)
    print(preds)
    if (preds[0])[0].item() >= 0.800000:
        result = "NORMAL"
    else:
        result = "COVID-19 POSITIVE"
    
    return result


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads folder
        img.save("./uploads/test/image.png")

        # Make prediction on uploaded saved image
        result = model_predict(img, model)

        # Display the result
        return jsonify(result=result)

    return None


if __name__ == '__main__':
    
    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
