from flask import Flask, jsonify, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

classifier = pickle.load(open('data/model.pkl', 'rb'))


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    result = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

    iris_data = request.form
    iris_data_df = pd.DataFrame(iris_data, index=[0])

    iris_prediction = classifier.predict(iris_data_df)

    return jsonify({'Species': result[iris_prediction.item()]})


if __name__ == '__main__':
    app.run(debug=True)
