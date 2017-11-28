from chalice import Chalice
from chalice import BadRequestError, UnauthorizedError
import json
import pickle
import logging
import requests

# Specific for importing sklearnin lambda
# see https://github.com/ryansb/sklearn-build-lambda
import os
import ctypes

# Use ctypes to support C data types, required for sklearn and numpy
for d, _, files in os.walk('lib'):
    for f in files:
        if f.endswith('.a'):
            continue
        ctypes.cdll.LoadLibrary(os.path.join(d, f))

import sklearn
import numpy as np


app = Chalice(app_name='demo')
app.debug = True
app.log.setLevel(logging.DEBUG)


@app.route("/")
def index():
    return {"hello": "world"}

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in app.current_request.json_body:
        raise UnauthorizedError("The image attribute is required.")

    image = json.loads(app.current_request.json_body['image'])

    if len(image) != 64:
        raise BadRequestError("Image should be serialized 8x8 grayscale. It's {}.".format(len(image)))

    model = pickle.load(open('chalicelib/model.pkl', 'rb'))
    return {"class": model.predict([image]).tolist()}

@app.schedule('rate(1 minute)')
def log_ip(event):
    r = requests.get('http://httpbin.org/ip').json()

    # We will use the logs, but we could save in a DB.
    app.log.debug("IP: " + r['origin'])
    return {"status": "ok"}
