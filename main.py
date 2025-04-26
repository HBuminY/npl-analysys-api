
from flask import Flask, request, jsonify
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download required NLTK data
nltk.download('vader_lexicon')

app = Flask(__name__)

def analyze_emotion(text):
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    
    compound = scores['compound']
    intensity = abs(compound)
    
    if compound >= 0.5:
        return "happy"
    elif compound <= -0.5:
        return "angry"
    elif -0.5 < compound < 0.5 and intensity >= 0.2:
        return "surprised"
    else:
        return "sad"

@app.route('/')
def hello():
    return "Hello World!"

@app.route('/emotion', methods=['POST'])
def get_emotion():
    print(f"Received emotion analysis request with text: {request.json.get('text', '')}")
    text = request.json.get('text', '')
    emotion = analyze_emotion(text)
    return jsonify({"result": emotion})

@app.before_request
def log_request():
    print(f"Incoming {request.method} request to {request.path}")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
