from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
from io import BytesIO
import numpy as np
import os
from flask_cors import CORS
import io

app = Flask(__name__)
CORS(app)
model = load_model('drunk_or_not_model.h5')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    img_bytes = BytesIO(file.read())
    img = Image.open(img_bytes)
    img = img.resize((150, 150))  
    img = img.convert('RGB')      

    img_array = np.array(img) / 255.0  
    img_array = img_array.reshape((1, 150, 150, 3)) 

    prediction = model.predict(img_array)
    result = 'sober' if prediction[0][0] > 0.5 else 'drunk'

    return jsonify({'prediction': result})



if __name__ == '__main__':
   app.run( debug=True,use_reloader=False)