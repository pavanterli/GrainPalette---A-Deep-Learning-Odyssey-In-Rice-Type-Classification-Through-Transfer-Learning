from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("rice_grain_classifier.h5")
class_names = ["Rice Type 1", "Rice Type 2", "Rice Type 3", "Rice Type 4", "Rice Type 5"]

def predict_rice(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return class_names[np.argmax(prediction)]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            file_path = os.path.join("static", file.filename)
            file.save(file_path)
            result = predict_rice(file_path)
            return render_template("index.html", image=file_path, result=result)
    return render_template("index.html", image=None, result=None)

if __name__ == "__main__":
    app.run(debug=True)
