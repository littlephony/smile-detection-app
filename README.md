# smile-detection-app

![workflow badge](https://github.com/littlephony/smile-detection-app/actions/workflows/build.yml/badge.svg)

A simple Flask app serving a PyTorch model for smile classification.

## Table of contents

* [General info](#general-info)
* [Technology](#technology)
* [Setup](#setup)
* [How to use](#how-to-use)
* [Planned changes](#planned-changes)
* [Sources and inspiration](#sources-and-inspiration)

## General info

I trained a convolutional neural network to recognize smiling faces using the CelebA dataset. To serve it, I used the Flask web-framework.

## Technology

- `PyTorch` to train the model
- `Flask` to serve the model
- `Black` to format Python code

## Setup

1. Clone the repository

  ```
  $ git clone https://github.com/littlephony/language-modeling-app.git
  ```

2. Install dependencies specified in `requirements.txt` in `server` directory:

  ```
  $ cd server
  $ pip install -r requirements.txt
  ```

3. To run the application you can simply run `app.py` file located in the `server` directory:

  ```
  $ python app.py
  ```
  
## How to use

To use the application, you should send an image file via a `POST` request to the `/api/predict` endpoint:

```Python
import requests

resp = requests.post(
    "http://localhost:5000/api/predict",
    files={'file': open('path/to/image/img.jpg', 'rb')}
)

print(resp.content)
```

## Planned changes

I have the following changes planned for this project:

1. Add more classification models 
2. Add UI to make the application easier to use.

## Sources and inspiration

As I was developing this project I have drawn insight and inspiration from the following sources:

- [Machine Learning with PyTorch and Scikit-Learn by Sebastian Raschka](https://www.packtpub.com/product/machine-learning-with-pytorch-and-scikit-learn/9781801819312#_ga=2.34450529.1118007180.1679687705-725254805.1672444559)
- [Deploying PyTorch in Python via a REST API with Flask](https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html)
