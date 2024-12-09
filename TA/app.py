import os
import torch
import timm
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

# Inisialisasi Flask
app = Flask(__name__)

# Konfigurasi path dan ekstensi file yang diperbolehkan
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Pastikan folder upload ada
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Memuat model yang sudah diekstrak
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = timm.create_model('inception_resnet_v2', pretrained=False, num_classes=6)  # Update num_classes menjadi 6
model.load_state_dict(torch.load('checkpoint.pt', map_location=device), strict=False)
model.to(device)
model.eval()  # Setel model ke mode evaluasi

# Fungsi untuk mengecek ekstensi file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Fungsi untuk melakukan prediksi pada gambar
def predict_image(image_path):
    # Transformasi gambar untuk input ke model
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Inception model membutuhkan ukuran 299x299
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # Inferensi
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    # Mengembalikan label kelas yang diprediksi
    return predicted.item()

# Fungsi untuk mendapatkan nama label berdasarkan indeks kelas
def get_label_name(label_idx):
    # Daftar nama kelas, sesuaikan dengan jumlah dan urutan kelas dalam model
    label_names = ['Calculus', 'Data caries', 'Gingivitis', 'Tooth Discoloration', 'Mouth Ulcer', 'Hypodontia']
    return label_names[label_idx]

# Route utama untuk menampilkan halaman upload gambar
@app.route('/')
def index():
    return render_template('index.html')

# Route untuk menerima gambar upload dan memberikan hasil prediksi
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Prediksi gambar yang di-upload
            label_idx = predict_image(filepath)
            label_name = get_label_name(label_idx)
            
            # Tampilkan gambar dan hasil prediksi
            return render_template('index.html', filename=filename, label=label_name)
        except Exception as e:
            return f"Error dalam proses prediksi: {str(e)}"
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)