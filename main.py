import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)
model = joblib.load('randomForestRegressor.pkl')  # ← using joblib now

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    return render_template('home.html', prediction_text=f"AQI for Jaipur {prediction[0]:.2f}")

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    return jsonify(prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
