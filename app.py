"""
Super Resolution Web App — Flask Backend
Run: python3 app.py
Then open: http://localhost:5000
"""

from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import os
import io
import base64
import socket
from pathlib import Path

app = Flask(__name__, static_folder=".", static_url_path="")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
OUTPUT_FOLDER = BASE_DIR / "outputs"
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

MODEL_PATHS = {
    "fsrcnn": "/Users/pranesh/Desktop/Super-resolution with deep learning/FSRCNN_x4.pb",
    "edsr": "/Users/pranesh/Desktop/Super-resolution with deep learning/EDSR_x4.pb",
}

def encode_image(img):
    """Convert OpenCV image to base64 PNG string for sending to browser."""
    _, buffer = cv2.imencode(".png", img)
    return base64.b64encode(buffer).decode("utf-8")


@app.route("/")
def index():
    return send_file(BASE_DIR / "index.html")


@app.route("/enhance", methods=["POST"])
def enhance():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    model_choice = request.form.get("model", "fsrcnn")
    scale = int(request.form.get("scale", 4))
    model_path = MODEL_PATHS.get(model_choice, MODEL_PATHS["fsrcnn"])

    if not os.path.exists(model_path):
        return jsonify({
            "error": f"Model file '{model_path}' not found. Make sure it's in the same folder as app.py."
        }), 400

    file = request.files["image"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Could not read image. Make sure it's a valid JPG or PNG."}), 400

    # Bicubic upscale
    h, w = img.shape[:2]
    bicubic = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    # AI Super Resolution
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(str(model_path))
        sr.setModel(model_choice, scale)
        result = sr.upsample(img)
    except Exception as e:
        return jsonify({"error": f"Super resolution failed: {str(e)}"}), 500

    return jsonify({
        "original":  encode_image(img),
        "bicubic":   encode_image(bicubic),
        "sr_result": encode_image(result),
        "orig_size":  f"{w}×{h}px",
        "out_size":   f"{w*scale}×{h*scale}px",
    })


if __name__ == "__main__":
    # Port 5000 can be occupied by system services on macOS. Pick a free one.
    preferred_port = int(os.getenv("PORT", "5001"))
    port = preferred_port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if sock.connect_ex(("127.0.0.1", preferred_port)) == 0:
            port = 5002

    print("\n  Super Resolution Web App")
    print(f"  Open http://127.0.0.1:{port} in your browser\n")
    app.run(debug=True, host="127.0.0.1", port=port)
