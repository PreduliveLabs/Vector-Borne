import tensorflow as tf
import joblib

CNN=joblib.load("CNN")


import webview
from flask import Flask, render_template, request, url_for,send_from_directory,Response
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.preprocessing import image
import os


import pandas as pd
df=pd.read_csv(r"city_diseases.csv")





app = Flask(__name__)
app.config["SECRET_KEY"] = 'ajashjkjm'
app.config['UPLOAD_FOLDER'] = "uploads"
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}


webview.create_window("hello",app)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']



list1=['Visakhapatnam','Guntur','Lucknow', 'Varanasi']


@app.route('/')
def home():
    return render_template("main_page.html")
def re_size(filepath):
    img = image.load_img(filepath,target_size=(250,250))
    img_array = image.img_to_array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values between 0 and 1
    val1 = CNN.predict(img_array)  # Assuming loaded_model is defined elsewhere
    val1 = np.argmax(val1)
    return val1


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/file', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_url = url_for('uploaded_file', filename=filename)
            img_path="uploads\\"+filename
            prediction=re_size(img_path)

            return render_template('main_page.html', prediction=list1[prediction],file_url=file_url,disease=list(df[df["City"]==list1[prediction]].Diseases)[0])

    return render_template('main_page.html', prediction=None,file_url=None,disease=None)


if __name__ == '__main__':
    webview.start()

