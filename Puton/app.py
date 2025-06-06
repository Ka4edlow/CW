from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image
from main import NeuralNetwork
import tensorflow as tf
import pickle
from tf_wrapper import TFModelWrapper

app = Flask(__name__)
app.config['UPLOAD_DIR'] = 'static/uploads'

model_mlp_numpy = NeuralNetwork(input_dim=32*32*3, hidden_dim=128, output_dim=10)
model_mlp_numpy.load('mlp_cifar10_best_model.pkl')

with open('mlp_cifar10_best_model_tensorflow.pkl', 'rb') as f:
    model_tensorflow = pickle.load(f)

labels = ['літак', 'автомобіль', 'птах', 'кіт', 'олень',
          'собака', 'жаба', 'кінь', 'корабель', 'вантажівка']

def prepare_image(img_path):
    image = Image.open(img_path).resize((32, 32)).convert('RGB')
    img_array = np.asarray(image, dtype=np.float32) / 255.0
    return img_array.reshape(-1, 32*32*3).T, img_array.reshape(1, 32, 32, 3)

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    prediction_result = None
    uploaded_filename = None
    model_choice = 'numpy'

    if request.method == 'POST':
        model_choice = request.form.get('model', 'numpy')
        uploaded_file = request.files.get('image')

        if uploaded_file and uploaded_file.filename:
            uploaded_filename = secure_filename(uploaded_file.filename)
            full_path = os.path.join(app.config['UPLOAD_DIR'], uploaded_filename)
            uploaded_file.save(full_path)

            img_numpy_format, img_tensorflow_format = prepare_image(full_path)

            if model_choice == 'tensorflow':
                pred = model_tensorflow.predict(img_tensorflow_format)
                predicted_class = np.argmax(pred, axis=1)[0]
            else:
                predicted_class = model_mlp_numpy.predict(img_numpy_format)[0]

            prediction_result = labels[predicted_class]
        else:
            prediction_result = "Файл не обрано або порожній."

    return render_template('index.html', prediction=prediction_result,
                           image=uploaded_filename, selected_model=model_choice)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_DIR'], exist_ok=True)
    app.run(debug=True)