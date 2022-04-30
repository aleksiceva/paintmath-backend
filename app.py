import json

from flask import Flask, jsonify, request
from flask_cors import CORS

from model.model import classify

app = Flask(__name__)
CORS(app)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = json.loads(request.get_data().decode()).get("image")
        class_name = classify(file)
        response = jsonify({'class_name': class_name})
        return response


if __name__ == '__main__':
    app.run()
