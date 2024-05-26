import numpy as np
from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from PIL import Image

app = Flask(__name__)

# Load the trained model
model = load_model('trained_models/my_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    img_file = request.files['file']
    img = Image.open(img_file)
    img = img.resize((64, 64))
    img_array = np.array(img)

    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)

    # Check the prediction value
    if predictions[0][0] > 0.5:
        output = "Coronal"
    else:
        output = "Axial"

    return jsonify({'prediction': output})

if __name__ == '__main__':
    app.run(debug=True)
