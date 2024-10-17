from flask import Flask, render_template;
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/profile')
def profil():
    return render_template('profil_raffi.html')

@app.route('/contact')
def contact():
    return 'hubungi kami'

if __name__ == '__main__':
    app.run(debug=True)