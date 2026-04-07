import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def index():
    return render_template('index.html')\

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

if __name__ == '__main__':
    app.run(debug=True)