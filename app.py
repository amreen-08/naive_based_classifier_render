import numpy as np
from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

app = Flask(__name__)

# Load and train model
iris = load_iris()
X = iris.data
Y = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=1
)

model = GaussianNB()
model.fit(X_train, Y_train)


@app.route("/")
def home():
    return "Iris Model API is Running 🚀"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)

    return jsonify({
        "prediction": int(prediction[0])
    })


if __name__ == "__main__":
    app.run(debug=True)