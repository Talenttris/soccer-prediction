from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model
model = joblib.load("soccer_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get input data
    data = request.json
    input_data = pd.DataFrame([data])

    # Make prediction
    prediction = model.predict(input_data)
    outcome = {1: "Home Win", 0: "Draw", -1: "Away Win"}[prediction[0]]

    return jsonify({"prediction": outcome})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
