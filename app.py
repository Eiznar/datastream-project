import flask
import pickle
import pandas as pd
from flask import jsonify
from models.utils import clean_text
import json
import numpy as np

from configparser import ConfigParser, RawConfigParser

configur = RawConfigParser()
configur.read('./config.ini')
port = configur.getint('Application','port')

pkl_filename = "models/logreg_model.pkl"
with open(pkl_filename, 'rb') as file:
    model = pickle.load(file)
print(model)

app = flask.Flask(__name__)

@app.route('/', methods=['GET'])
def predict():
    """
    Function to predict whether the tweet is fake news or not.
    """
    # json in the form {'tweet': str}
    tweet = flask.request.args.get('tweet')
    print(tweet)
    tweet = clean_text(tweet)
    x_ = pd.Series(np.array([tweet]))
    output = model.predict_proba(x_)
    fake = float(output[0][0])
    real = float(output[0][1])
    if fake > real:
        pred = "fake"
    else:
        pred = "real"
    return jsonify({'fake': fake, 'real': real, "prediction": pred})

if __name__ == '__main__':
    app.run(port=port, debug=True)
