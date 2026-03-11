from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("safety_model.pkl")
encoders = joblib.load("encoders.pkl")

@app.route("/", methods=["GET","POST"])
def index():

    prediction = ""

    if request.method == "POST":

        lighting = request.form["lighting"]
        crowd = request.form["crowd"]
        incidents = int(request.form["incidents"])
        time = request.form["time"]

        lighting_enc = encoders["lighting"].transform([lighting])[0]
        crowd_enc = encoders["crowd"].transform([crowd])[0]
        time_enc = encoders["time"].transform([time])[0]

        features = np.array([[lighting_enc,crowd_enc,incidents,time_enc]])

        pred = model.predict(features)

        prediction = encoders["safety"].inverse_transform(pred)[0]

    return render_template("index.html",prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)