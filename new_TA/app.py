import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Inisialisasi Flask app
app = Flask(__name__)

# Load model .keras
model = load_model('Model_MobileNetV2.keras')

# Path folder untuk template
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Fungsi untuk memproses dan memprediksi gambar
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Ukuran gambar yang digunakan oleh model
    img_array = img_to_array(img)  # Mengubah gambar ke array
    img_array = np.expand_dims(img_array, axis=0)  # Menambah dimensi batch
    img_array = preprocess_input(img_array)  # Preprocessing sesuai MobileNetV2
    
    # Prediksi menggunakan model
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]  # Ambil kelas yang diprediksi
    return predicted_class

# Route untuk homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route untuk menerima gambar dan mengklasifikasikan
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        # Simpan file gambar yang diupload
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Prediksi kelas gambar
        predicted_class = predict_image(filepath)
        
        # Kembalikan hasil prediksi
        class_labels = ['Calculus', 'Data caries', 'Gingivitis', 'Tooth Discoloration', 'Mouth Ulcer', 'Hypodontia']  # Gantilah dengan label kelas yang sesuai
        predicted_label = class_labels[predicted_class]
        
        return jsonify({'predicted_class': predicted_label})

# Menjalankan Flask app
if __name__ == "__main__":
    app.run(debug=True, port=8000)
    

