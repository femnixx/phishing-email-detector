import joblib
from flask import Flask, request, jsonify, render_template
import sqlite3

app = Flask(__name__)

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def init_db():
    conn = sqlite3.connect('feedback.db')
    conn.execute('''
                 CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email_text TEXT,
                    prediction TEXT,
                    was_correct BOOLEAN,
                    status TEXT DEFAULT 'pending',
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                 )
                 ''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    email_text = data.get('text', '')

    if not email_text.strip():
        return jsonify({'error': 'No text provided'}), 400
    
    vec = vectorizer.transform([email_text])
    guess = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]

    return jsonify({
        'result': 'PHISHING' if guess == 1 else 'SAFE',
        'probability': round(prob[1] * 100, 2)
    })

@app.route('/feedback', methods=['POST'])
def feedback(): 
    data = request.get_json()
    conn = sqlite3.connect('feedback.db')
    conn.execute(
        'INSERT INTO feedback (email_text, prediciton, was_correct) VALUES (?, ?, ?)', (data['text'], data['prediction'], data['was_correct'])
    )
    conn.commit()
    conn.close()
    return jsonify({ 'status': 'ok' })

@app.route('/admin')
def admin():
    conn = sqlite3.connect('feedback.db')
    rows = conn.execute('SELECT * FROM feedback').fetchall()
    conn.close()
    return render_template('admin.html', rows=rows)


if __name__ == '__main__':
    app.run(debug=True)