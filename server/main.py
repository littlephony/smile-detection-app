from flask import Flask
import torch

from model import SmileClassificationNet


app = Flask(__name__)
model = SmileClassificationNet()
model.load_state_dict(torch.load('../model/model.pt', map_location='cpu'))


@app.route('/')
def hello():
    return 'Hello, World!'