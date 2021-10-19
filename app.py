from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_restful import Api, Resource, reqparse
from flask import request

app = Flask(__name__)
model = pickle.load(open("binary_classification_model.pkl", "rb"))
print("HIIIIIIIIII")
print(model)

@app.route('/')
def data():
    return "Give /predict in URL"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(data["parameter"])])
    print("****", prediction)
    output = prediction[0]
    print("Prediction done")
    return jsonify(output)

if __name__ == "__main__":
	app.run('0.0.0.0', port=5000, debug=True)
