# AI Super Resolution Web App

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black?logo=flask)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Enhance low-resolution images using deep learning-based super resolution with an interactive Flask web interface. Upload an image, choose a model, and compare original, bicubic, and AI-enhanced results side by side.

## Screenshot

> Add your app screenshot here after deployment/testing.

![App Screenshot Placeholder](https://via.placeholder.com/1200x700.png?text=AI+Super+Resolution+Web+App)

## Features

- Drag-and-drop image upload UI
- Supports `FSRCNN` and `EDSR` super-resolution models
- Side-by-side comparison: Original vs Bicubic vs AI Super Resolution
- Download processed results directly from the browser
- Clean Flask backend with OpenCV DNN SuperRes pipeline

## How It Works (FSRCNN Explained Simply)

FSRCNN (Fast Super-Resolution Convolutional Neural Network) is a lightweight neural network designed to upscale images quickly.

In simple terms:

1. The model looks at small patterns in the low-resolution image (edges, textures, details).
2. It learns how those patterns should appear in a higher-resolution version.
3. Instead of just stretching pixels (like normal resizing), it predicts missing details to produce a sharper result.

That is why FSRCNN is usually much better than basic interpolation methods like bicubic scaling.

## Tech Stack

- **Python** - core application logic
- **OpenCV** - image processing and super-resolution model inference
- **Flask** - backend web server and API endpoints
- **HTML, CSS, JavaScript** - frontend interface and interactions
- **Matplotlib** - used in local script-based comparison workflows

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/pranesh236/ai-super-resolution.git
cd ai-super-resolution
```

2. **Create and activate a virtual environment (recommended)**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies**

```bash
pip install flask opencv-contrib-python numpy matplotlib
```

4. **Download model files**

- FSRCNN model (`FSRCNN_x4.pb`)
- EDSR model (`EDSR_x4.pb`)

Place them in the project directory (or update model paths in `app.py` if needed).

5. **Run the app**

```bash
python3 app.py
```

## Usage

1. Start the Flask server:

```bash
python3 app.py
```

2. Open the shown local URL in your browser (for example `http://127.0.0.1:5001` or `http://127.0.0.1:5002`).
3. Drag and drop (or browse) an image file.
4. Select a model (`FSRCNN` or `EDSR`) and scale factor.
5. Click **Generate Super Resolution**.
6. Review the side-by-side panels and download any result.

## Project Structure

```text
ai-super-resolution/
├── app.py
├── index.html
├── super_resolution.py
├── README.md
├── .gitignore
├── uploads/
└── outputs/
```

## Models: FSRCNN vs EDSR

### FSRCNN
- Faster inference, suitable for near real-time demos
- Lightweight architecture
- Great balance of speed and quality

### EDSR
- Higher visual quality in many cases
- Better recovery of fine details and textures
- Heavier model, slower than FSRCNN

If you prioritize speed, use **FSRCNN**.  
If you prioritize best visual quality and can accept longer processing time, use **EDSR**.

## Author

**pranesh236**

## License

This project is licensed under the **MIT License**.