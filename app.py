import flask
import pickle
import pandas as pd
from flask import jsonify
from functions import clean_text
import json

pkl_filename = "models/logreg_model.pkl"
with open(pkl_filename, 'rb') as file:
    model = pickle.load(file)
print(model)

app = flask.Flask(__name__)

@app.route('/', methods=['GET'])
def predict():
    # json in the form {'tweet': str}
    tweet = flask.request.args.get('tweet')
    tweet = clean_text(tweet)
    x_df = pd.DataFrame.from_dict({"tweet": [tweet]})
    output = model.predict_proba(x_df)
    print(output)
    fake = float(output[0][0])
    real = float(output[0][1])
    if fake > real:
        pred = "fake"
    else:
        pred = "real"
    print(type(pred))
    return jsonify({'fake': fake, 'real': real, "prediction": pred})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
