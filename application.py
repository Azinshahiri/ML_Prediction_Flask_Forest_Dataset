from flask import Flask, render_template, request,jsonify
import pickle
import numpy as np
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

# =========================Load all models =================
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/elastic_net.pkl", "rb") as f:
    elastic_net_model = pickle.load(f)

with open("models/elastic_net_cv.pkl", "rb") as f:
    elastic_net_cv_model = pickle.load(f)

with open("models/lasso_cv.pkl", "rb") as f:
    lasso_model_cv_model = pickle.load(f)

with open("models/lasso_model.pkl", "rb") as f:
    lasso_model = pickle.load(f)

with open("models/ridge_cv.pkl", "rb") as f:
    ridge_model = pickle.load(f)

@app.route('/')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        #====================================== Get input values and convert to float===================================
        features = [float(x) for x in request.form.values() if x != request.form.get("model_choice")]
        input_array = np.array([features])
        input_scaled = scaler.transform(input_array)

        #====================================== Choose model=======================
        selected_model = request.form.get("model_choice")
        if selected_model == "linear":
            prediction = model.predict(input_scaled)
        elif selected_model == "lasso":
            prediction = lasso_model.predict(input_scaled)
        elif selected_model == "lasso_cv":
            prediction = lasso_model_cv_model.predict(input_scaled)
        elif selected_model == "ridge":
            prediction = ridge_model.predict(input_scaled)
        elif selected_model == "elastic":
            prediction = elastic_net_model.predict(input_scaled)
        elif selected_model == "elastic_cv":
            prediction = elastic_net_cv_model.predict(input_scaled)
        else:
            return render_template('index.html', prediction="Invalid model selected")

        return render_template('index.html', prediction=f"Prediction: {prediction[0]:.2f}")

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")



if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)

