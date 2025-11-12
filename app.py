import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

model = joblib.load('fake_job_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/')
def home():
    return "Fake Job Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get('description', '')
    if not text:
        return jsonify({"error": "No job description provided"}), 400
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0, 1]
    return jsonify({
        "prediction": "Fake Job" if pred == 1 else "Real Job",
        "probability_fake": round(float(proba), 4)
    })

if __name__ == '__main__':
    app.run(debug=True)