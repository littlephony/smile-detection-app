import io

from flask import Flask, url_for, redirect, jsonify, request
import torch
from torchvision import transforms
from PIL import Image
from model import SmileClassificationNet


app = Flask(__name__)
model = SmileClassificationNet()
model.load_state_dict(torch.load('../model/model.pt', map_location='cpu'))


def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(178),
        transforms.Resize(64),
        transforms.ToTensor(),
    ])

    image = Image.open(io.BytesIO(image_bytes))

    return transform(image).unsqueeze(0)


def get_prediction(image_bytes):
    image_tensor = transform_image(image_bytes)
    proba = model(image_tensor)
    return proba
    

@app.route('/')
def hello():
    return redirect(url_for('/api/predict'))


@app.route('/api/predict')
def predict():
    if request.method == 'POST':
        file = request.files['file']
        image_bytes = file.read()
        output = get_prediction(image_bytes)

        return jsonify({'smile': output.item() > 0.5,
                        'proba': output.item()})
    
if __name__ == '__main__':
    app.run()
