from flask import Flask, render_template, request
import joblib
import os, csv
from datetime import datetime
import pandas as pd

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('fake_job_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Global counters and last prediction cache
fake_count = 0
real_count = 0
last_label = None
last_confidence = None
last_text = None

LOG_PATH = 'predictions_log.csv'


@app.route('/')
def home():
    return render_template(
        'index.html',
        fake=fake_count,
        real=real_count,
        last_label=last_label,
        last_confidence=last_confidence,
        last_text=last_text
    )


def append_log(description, label, confidence):
    file_empty = (not os.path.isfile(LOG_PATH)) or os.path.getsize(LOG_PATH) == 0
    with open(LOG_PATH, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        if file_empty:
            w.writerow(['timestamp', 'job_description', 'prediction', 'confidence'])
        w.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), description, label, confidence])


@app.route('/history')
def history():
    if not os.path.isfile(LOG_PATH) or os.path.getsize(LOG_PATH) == 0:
        return render_template('history.html', rows=[], headers=['timestamp', 'job_description', 'prediction', 'confidence'])
    try:
        df = pd.read_csv(LOG_PATH)
    except pd.errors.EmptyDataError:
        return render_template('history.html', rows=[], headers=['timestamp', 'job_description', 'prediction', 'confidence'])
    return render_template('history.html', rows=df.values.tolist(), headers=df.columns.tolist())


@app.route('/predict', methods=['POST'])
def predict():
    global fake_count, real_count, last_label, last_confidence, last_text
    job_desc = request.form.get('job_description', '').strip()

    # Validation
    if not job_desc:
        return render_template(
            'index.html',
            error="Description cannot be empty.",
            fake=fake_count, real=real_count,
            last_label=last_label, last_confidence=last_confidence, last_text=last_text
        )
    if len(job_desc.split()) < 5:
        return render_template(
            'index.html',
            error="Please enter at least 5 words.",
            fake=fake_count, real=real_count,
            last_label=last_label, last_confidence=last_confidence, last_text=last_text
        )
    letters = sum(c.isalpha() for c in job_desc)
    ratio = letters / max(1, len(job_desc))
    if ratio < 0.40:
        return render_template(
            'index.html',
            error="Input seems non-text (too many numbers/symbols).",
            fake=fake_count, real=real_count,
            last_label=last_label, last_confidence=last_confidence, last_text=last_text
        )

    # Predict
    X_input = vectorizer.transform([job_desc])
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][1]

    label = "Fake Job" if pred == 1 else "Real Job"
    confidence = round(prob * 100, 2) if pred == 1 else round((1 - prob) * 100, 2)

    if pred == 1:
        fake_count += 1
    else:
        real_count += 1

    last_label = label
    last_confidence = confidence
    last_text = job_desc

    append_log(job_desc, label, confidence)

    return render_template(
        'result.html',
        label=label,
        confidence=confidence,
        description=job_desc,
        fake=fake_count,
        real=real_count
    )


if __name__ == '__main__':
    app.run(debug=True)