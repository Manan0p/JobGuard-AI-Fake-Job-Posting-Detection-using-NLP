# JobGuard AI – Fake Job Posting Detection

A modern, production-ready web application that detects fraudulent job postings using Machine Learning. Built with Flask, SQLite, and a beautiful responsive UI with dark/light theme support, admin dashboard, and real-time analytics.

**🌐 Live Demo:** https://jobguard-ai.up.railway.app/

---

## 🎯 What This Project Does

JobGuard AI helps job seekers identify scam postings by analyzing text patterns common in fraudulent listings:
- Exaggerated salary claims
- Vague job responsibilities
- Upfront payment requests
- Too-good-to-be-true offers

**Core Features:**
- ✅ Real-time job listing analysis with confidence scores
- ✅ Prediction history with search & filtering
- ✅ Admin dashboard with interactive charts & analytics
- ✅ Model retraining with custom datasets
- ✅ Dark/Light theme with persistent storage
- ✅ Session-based authentication for admins
- ✅ SQLite database for audit trails
- ✅ Responsive design (mobile, tablet, desktop)
- ✅ Modern UI with smooth animations

---

## 👥 Who Is This For?

Perfect for learning:
- **Data Science:** Text preprocessing, TF-IDF vectorization, model training/evaluation
- **Web Development:** Flask routing, Jinja2 templating, session management
- **Database Design:** SQLite schema, CRUD operations, time-series logging
- **Frontend:** Modern CSS (variables, grid, flexbox), JavaScript interactivity, theme switching
- **DevOps:** Project structure, requirements management, database migrations

---

## 📁 Project Structure

```
Infosys-ISpringboard/
├── app.py                          # Flask server & routes
├── fake_job_pipeline.py            # ML training pipeline
├── fake_job_model.pkl              # Trained model (auto-generated)
├── tfidf_vectorizer.pkl            # TF-IDF vectorizer (auto-generated)
├── job_predictions.db              # SQLite database (auto-created)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── templates/
    ├── index.html                  # Public: Job analysis form
    ├── result.html                 # Public: Prediction results
    ├── home.html                   # Public: Landing page
    ├── history.html                # Public: Prediction history with filters
    ├── login.html                  # Admin: Login page
    ├── dashboard.html              # Admin: Analytics & model management
    └── retrain_logs.html           # Admin: Training history & charts
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.11+ 
- pip (comes with Python)

### 2. Install Dependencies
```bash
cd Infosys-ISpringboard
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Train the Model (First Time Only)
```bash
python fake_job_pipeline.py
```
This creates:
- `fake_job_model.pkl` – Trained Logistic Regression classifier
- `tfidf_vectorizer.pkl` – Text vectorizer
- Evaluation plots showing model performance

### 4. Start the Web App
```bash
python app.py
```
Open browser: **`http://127.0.0.1:5000/`**

---

## 🔐 Admin Access

**Default Credentials:**
- **Username:** `admin`
- **Password:** `password123`

⚠️ **Change these credentials in production!**

Edit `app.py` line with:
```python
if username == 'your_username' and password == 'your_secure_password':
```

Or use environment variables:
```python
import os
ADMIN_USER = os.getenv('ADMIN_USER', 'admin')
ADMIN_PASS = os.getenv('ADMIN_PASS', 'password123')
```

---

## 📖 Using the Interface

### 🏠 Public Pages (No Login Required)

**Home Page** (`/`)
- Welcome screen with stats (total predictions, fake/real counts)
- Quick links to analyze jobs
- Model accuracy info

**Analysis Page** (`/predict_form`)
- Paste job description (≥5 words, ≥40% alphabetic)
- Click "Analyze Now"
- Get instant prediction with confidence %

**History Page** (`/history`)
- View all past predictions (newest first)
- Search by job description
- Filter by type (All / Fake / Real)
- See accuracy stats dashboard

### 🔑 Admin Pages (Login Required)

**Dashboard** (`/admin_dashboard`)
- 📊 Total predictions, fake/real counts, model accuracy
- 📈 Line chart: Daily prediction volume
- 🎯 Pie chart: Fake vs Real distribution
- 🔄 Retrain section with drag-and-drop file upload
- ℹ️ Current model status & info
- 📋 Recent training logs

**Training Logs** (`/retrain_logs`)
- 📋 Complete training history with timestamps
- 📈 Accuracy trend line chart
- 🎯 Accuracy indicators (Excellent/Good/Fair)
- 🔍 Search logs by training source
- 📊 Performance summary & model status

---

## 🌐 API Routes

| Route | Method | Auth | Purpose |
|-------|--------|------|---------|
| `/` | GET | No | Landing page |
| `/predict_form` | GET | No | Analysis form |
| `/predict` | POST | No | Predict (returns JSON) |
| `/history` | GET | No | Prediction history |
| `/admin_login` | GET/POST | No | Admin login |
| `/admin_dashboard` | GET | Yes | Analytics & retraining |
| `/retrain_logs` | GET | Yes | Training history |
| `/retrain` | POST | Yes | Trigger retraining |
| `/logout` | GET | Yes | Logout & redirect |

---

## 💾 Database Schema

**`predictions` table:**
```sql
id (INTEGER) | job_description (TEXT) | prediction (TEXT) | confidence (REAL) | timestamp (DATETIME)
```

**`admin` table:**
```sql
id (INTEGER) | username (TEXT) | password (TEXT)
```

**`retrain_logs` table:**
```sql
id (INTEGER) | accuracy (REAL) | timestamp (DATETIME) | training_source (TEXT)
```

---

## 🎨 Features Showcase

### Design
- ✨ Modern gradient buttons & cards
- 🌓 Dark/Light theme (persistent across sessions)
- 📱 Fully responsive (mobile, tablet, desktop)
- 🎯 Smooth animations & transitions
- ♿ Semantic HTML & accessibility

### Interactivity
- 🔍 Real-time search & filtering
- 📊 Interactive Chart.js graphs
- 📁 Drag-and-drop file upload
- 🔄 Auto-stats calculation
- ⚡ Form validation & error handling

### Performance
- 💾 Lightweight SQLite (no server needed)
- ⚡ Instant predictions (< 100ms)
- 📦 Minimal dependencies
- 🚀 Ready to deploy

---

## 🔧 Configuration

### Change Timezone
Edit `app.py`:
```python
def format_time(timestamp_str):
    # For India (IST): UTC+5:30
    your_timezone = timezone(timedelta(hours=5, minutes=30))
    
    # Other options:
    # UTC: timezone(timedelta(hours=0))
    # US EST: timezone(timedelta(hours=-5))
    # Singapore: timezone(timedelta(hours=8))
    # UK: timezone(timedelta(hours=1))
```

### Change Model Accuracy Thresholds
Edit `dashboard.html` & `retrain_logs.html`:
```html
{% if log['accuracy'] >= 95 %}
  <span class="badge success">✓ Excellent</span>
{% elif log['accuracy'] >= 90 %}
  <span class="badge success">✓ Good</span>
```

---

## 📊 How Predictions Work

1. **Input:** User submits job description
2. **Validation:** Check minimum words & alphabetic content
3. **Vectorization:** Convert text to TF-IDF numerical features
4. **Prediction:** Logistic Regression classifier outputs probability
5. **Classification:** Apply threshold (0.5) to determine Fake/Real
6. **Storage:** Save to SQLite with timestamp & confidence
7. **Output:** Display result with confidence % to user

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'joblib'` | Run `pip install -r requirements.txt` |
| Model files missing | Run `python fake_job_pipeline.py` first |
| Database errors | Delete `job_predictions.db` and restart app |
| Wrong time display | Update timezone in `format_time()` function |
| Login redirects to wrong page | Clear browser cache/cookies |
| Charts not showing | Check browser console for JS errors |
| Unicode errors on Windows | Save all files as UTF-8 |

---

## 🚀 Deployment
**Currently deployed on Railway:**  
🌐 https://jobguard-ai.up.railway.app/
```

### Local Network
```bash
python app.py
# Access from other computers on same network:
# http://<your-ip>:5000/
```

### Docker Deployment
```bash
docker build -t jobguard-ai .
docker run -p 5000:8000 jobguard-ai
```

## 🔐 Security Notes

⚠️ **This is a demo/learning project. For production:**

- [ ] Hash passwords using `werkzeug.security.generate_password_hash()`
- [ ] Use environment variables for secrets
- [ ] Enable HTTPS/SSL
- [ ] Add CSRF protection with Flask-WTF
- [ ] Implement rate limiting
- [ ] Add input sanitization
- [ ] Use a real database (PostgreSQL, MySQL)
- [ ] Add audit logging for admin actions

---

## 📚 Learning Resources

- **Flask:** https://flask.palletsprojects.com/
- **scikit-learn:** https://scikit-learn.org/
- **SQLite:** https://www.sqlite.org/
- **Chart.js:** https://www.chartjs.org/
- **NLTK:** https://www.nltk.org/

---

## 🎓 What You'll Learn

✅ End-to-end ML pipeline (data → model → deployment)
✅ Text preprocessing & vectorization
✅ Model selection & evaluation
✅ Flask web framework & routing
✅ Database design & queries
✅ Session-based authentication
✅ Modern responsive CSS & JavaScript
✅ Time-series analytics & visualization

---

## 🔄 Extending the Project

### Easy Additions
- [ ] Add email notifications for high-risk postings
- [ ] Implement user accounts (not just admin)
- [ ] Export predictions to CSV
- [ ] Add confidence score breakdown
- [ ] Mobile app using Flutter/React Native

### Advanced Additions
- [ ] Switch to modern NLP (BERT, DistilBERT)
- [ ] Add feedback loop (users mark false positives)
- [ ] Implement A/B testing for models
- [ ] Real-time model drift detection
- [ ] Integrate with job boards (LinkedIn, Indeed APIs)
- [ ] Deploy as microservice with Docker & Kubernetes

---

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### MIT License Summary

**You are free to:**
- ✅ Use commercially
- ✅ Modify
- ✅ Distribute
- ✅ Use privately

**Under the condition:**
- ⚖️ Include original copyright notice and license text

**Limitations:**
- ❌ No liability or warranty

---

## 💡 Tips for Success

1. **Start Simple:** Understand the basic flow before customizing
2. **Test Manually:** Try different job descriptions to see how model reacts
3. **Monitor Logs:** Check timestamp logs to understand prediction patterns
4. **Experiment:** Retrain with different datasets to improve accuracy
5. **Share:** Show it to friends and get feedback on UI/UX
6. **Deploy:** Once confident, deploy to cloud for others to use

---

## 🙋 FAQ

**Q: Can I use my own training data?**
A: Yes! Modify `fake_job_pipeline.py` to load your CSV/dataset instead of hardcoded samples.

**Q: How accurate is the model?**
A: Depends on training data quality. Current model achieves ~93-97% accuracy (see dashboard).

**Q: Can I change the prediction threshold?**
A: Yes, edit `app.py` in the `/predict` route:
```python
label = "Fake Job" if prob > 0.6 else "Real Job"  # Change from 0.5
```

**Q: Is this production-ready?**
A: It's a great foundation! Add security hardening before real deployment.

**Q: How do I add more features?**
A: Modify `fake_job_pipeline.py` to include additional text features, then retrain.

---

**Made with ❤️ using Flask + ML + Modern UI Design**

Start analyzing fake jobs today! 🚀