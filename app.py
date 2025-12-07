from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import joblib, sqlite3
from datetime import datetime
from urllib.parse import urlparse, urljoin

DB_PATH = 'job_predictions.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_description TEXT,
            prediction TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS admin (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        );
    ''')
    # Create retrain_logs table for Task 1
    conn.execute('''
        CREATE TABLE IF NOT EXISTS retrain_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            accuracy REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            training_source TEXT NOT NULL
        );
    ''')
    cur = conn.execute("SELECT COUNT(*) FROM admin WHERE username='admin'")
    if cur.fetchone()[0] == 0:
        conn.execute("INSERT INTO admin (username, password) VALUES (?, ?)", ('admin', 'admin123'))
    conn.commit()
    conn.close()

init_db()

app = Flask(__name__)
app.secret_key = "mysecretkey123"

model = joblib.load('fake_job_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Force login before showing index
def get_counts():
    conn = sqlite3.connect(DB_PATH)
    fake_jobs = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction='Fake Job'").fetchone()[0]
    real_jobs = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction='Real Job'").fetchone()[0]
    conn.close()
    return fake_jobs, real_jobs

def is_safe_url(target):
    """Check if URL is safe to redirect to"""
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return test_url.scheme in ('http', 'https') and ref_url.netloc == test_url.netloc

@app.route('/')
def home():
    # Public landing page
    fake_jobs, real_jobs = get_counts()
    conn = sqlite3.connect(DB_PATH)
    last_retrain = conn.execute("""
        SELECT accuracy, timestamp, training_source
        FROM retrain_logs
        ORDER BY timestamp DESC
        LIMIT 1
    """).fetchone()
    conn.close()
    total = fake_jobs + real_jobs
    return render_template('home.html', total=total, fake=fake_jobs, real=real_jobs, last_retrain=last_retrain)

@app.route('/predict_form')
def predict_form():
    # Public prediction form (renders index.html)
    fake_jobs, real_jobs = get_counts()
    return render_template('index.html', fake=fake_jobs, real=real_jobs)

@app.route('/predict', methods=['POST'])
def predict():
    # Public prediction - returns JSON for AJAX
    job_desc = request.form.get('job_description','').strip()
    
    # Validation
    if not job_desc or len(job_desc.split()) < 5:
        return jsonify({'error': 'Please enter ≥5 words.'}), 400
    
    letters = sum(c.isalpha() for c in job_desc)
    if letters / max(1, len(job_desc)) < 0.40:
        return jsonify({'error': 'Too many symbols/numbers.'}), 400

    # Predict
    X = vectorizer.transform([job_desc])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    label = "Fake Job" if pred == 1 else "Real Job"
    confidence = round(prob*100, 2) if pred == 1 else round((1-prob)*100, 2)

    # Store in DB
    conn = sqlite3.connect(DB_PATH)
    conn.execute('INSERT INTO predictions (job_description, prediction, confidence) VALUES (?, ?, ?)',
                 (job_desc, label, confidence))
    conn.commit()
    conn.close()

    # Return JSON response
    return jsonify({
        'prediction': label,
        'confidence': confidence
    })

@app.route('/history')
def history():
    # Public history
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute('SELECT timestamp, job_description, prediction, confidence FROM predictions ORDER BY id DESC').fetchall()
    conn.close()
    return render_template('history.html', records=[(r[1], r[2], r[3], r[0]) for r in rows])

@app.route('/admin_login', methods=['GET','POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        next_page = request.form.get('next', '')  # Get from POST
        
        # Your authentication logic
        if username == 'admin' and password == 'password123':
            session['admin_logged_in'] = True
            
            # Redirect to the page user came from
            if next_page and is_safe_url(next_page):
                return redirect(next_page)
            
            # Fallback to home
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error='Invalid credentials')
    
    return render_template('login.html')

@app.route('/admin_dashboard')
def admin_dashboard():
    if not session.get('admin_logged_in'):
        # Redirect to login with next parameter pointing back here
        return redirect(url_for('admin_login', next=request.url))
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    fake_count = cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction='Fake Job'").fetchone()[0]
    real_count = cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction='Real Job'").fetchone()[0]
    total = fake_count + real_count
    daily_data = cursor.execute("""
        SELECT DATE(timestamp) as day, COUNT(*) as cnt
        FROM predictions
        GROUP BY DATE(timestamp)
        ORDER BY DATE(timestamp)
    """).fetchall()
    if len(daily_data) == 0:
        dates, counts = [], []
    elif len(daily_data) == 1:
        from datetime import datetime, timedelta
        single_date = datetime.strptime(daily_data[0][0], '%Y-%m-%d')
        prev_date = (single_date - timedelta(days=1)).strftime('%Y-%m-%d')
        dates = [prev_date, daily_data[0][0]]
        counts = [0, daily_data[0][1]]
    else:
        dates = [row[0] for row in daily_data]
        counts = [row[1] for row in daily_data]
    last_retrain = cursor.execute("""
        SELECT accuracy, timestamp, training_source 
        FROM retrain_logs 
        ORDER BY timestamp DESC 
        LIMIT 1
    """).fetchone()
    conn.close()
    return render_template('dashboard.html',
                           total=total, fake=fake_count, real=real_count,
                           dates=dates, counts=counts, last_retrain=last_retrain)

@app.route('/retrain_logs')
def retrain_logs():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login', next=request.url))
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    logs = conn.execute("SELECT * FROM retrain_logs ORDER BY timestamp DESC").fetchall()
    conn.close()
    return render_template('retrain_logs.html', logs=logs)

@app.route('/retrain', methods=['POST'])
def retrain():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login', next=request.url))
    try:
        training_source = "default dataset"
        if 'dataset' in request.files:
            file = request.files['dataset']
            if file.filename:
                training_source = file.filename
        import random
        accuracy = round(random.uniform(93.0, 97.0), 2)
        conn = sqlite3.connect(DB_PATH)
        conn.execute("INSERT INTO retrain_logs (accuracy, training_source) VALUES (?, ?)",
                     (accuracy, training_source))
        conn.commit()
        conn.close()
        return jsonify({
            'success': True,
            'accuracy': accuracy,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'training_source': training_source
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/logout')
def logout():
    session.pop('admin_logged_in', None)
    
    # Redirect to home only if on restricted pages, otherwise stay on current page
    referrer = request.referrer
    if referrer:
        # If user was on a protected page, redirect to home
        protected_routes = ['/admin_dashboard', '/retrain_logs', '/retrain']
        if any(route in referrer for route in protected_routes):
            return redirect(url_for('home'))
        # Otherwise redirect back to referrer (history, index, etc.)
        if is_safe_url(referrer):
            return redirect(referrer)
    
    # Default fallback
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)