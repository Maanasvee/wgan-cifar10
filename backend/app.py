from flask import Flask, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import io

app = Flask(__name__)

generator = tf.keras.models.load_model("../models/generator.h5")

@app.route("/generate")

def generate():

    noise = np.random.normal(0,1,(1,100))

    img = generator.predict(noise)[0]

    img = (img*127.5 + 127.5).astype("uint8")

    image = Image.fromarray(img)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    img_str = base64.b64encode(buffer.getvalue()).decode()

    return jsonify({"image":img_str})

if __name__ == "__main__":
    app.run(debug=True)
