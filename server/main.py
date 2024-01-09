import cv2
import os

from flask import Flask
from flask import request
from flask import send_file
from src.preprocess import predict
from werkzeug.utils import secure_filename

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return "OK", 200


# Allows client to upload and receive predictions from images
@app.route("/predict", methods=["POST"])
def upload():
    if request.method == "POST":
        f = request.files["file"]

        # Saves the image to uploads folder
        path = f"./uploads/{secure_filename(f.filename)}"
        f.save(path)

        # Performs preprocessing to the image
        filename = "output.jpg"
        prediction = predict(path)
        cv2.imwrite(
            filename=f"./predictions/{secure_filename(filename)}", img=prediction
        )

        # Prepare file to send
        file_to_send = os.path.join("predictions", filename)

        return send_file(file_to_send, mimetype="image/jpg")
    else:
        return 405


if __name__ == "__main__":
    app.run(host="localhost", port=5000)
