from flask import Flask, request, jsonify, render_template
import joblib
import os

app = Flask(__name__)  # Fixed from _name_

# Load model and encoders
model = joblib.load("fraud_model.pkl")
merchant_encoder = joblib.load("merchant_encoder.pkl")
trans_encoder = joblib.load("trans_encoder.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        cc_num = int(data.get('cc_num'))
        merchant = data.get('merchant')
        trans_num = data.get('trans_num')

        merchant_encoded = merchant_encoder.transform([merchant])[0]
        trans_encoded = trans_encoder.transform([trans_num])[0]

        features = [[cc_num, merchant_encoded, trans_encoded]]
        prediction = model.predict(features)[0]

        return jsonify({'fraud': bool(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
