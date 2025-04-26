
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
    pos = scores['pos']
    neg = scores['neg']
    neu = scores['neu']

    text_lower = text.lower()

    # Keyword-based hints
    happy_keywords = ["love", "great", "awesome", "fantastic", "joy", "glad", "good"]
    sad_keywords = ["sad", "down", "depressed", "upset", "cry", "unhappy"]
    angry_keywords = ["angry", "mad", "furious", "annoyed", "hate", "irritated"]
    surprised_keywords = ["wow", "surprised", "unexpected", "shocked", "amazed", "astonished"]

    # Check keywords first
    for word in happy_keywords:
        if word in text_lower:
            return "happy"
    for word in sad_keywords:
        if word in text_lower:
            return "sad"
    for word in angry_keywords:
        if word in text_lower:
            return "angry"
    for word in surprised_keywords:
        if word in text_lower:
            return "surprised"

    # If no keyword found, fallback to sentiment
    if compound >= 0.5 and pos > 0.5:
        return "happy"
    elif compound <= -0.5 and neg > 0.5:
        return "angry"
    elif -0.3 < compound < 0.3 and neu > 0.6:
        return "surprised"
    else:
        return "sad"


@app.route('/')
def hello():
    return "Hello World!"

@app.route('/emotion', methods=['POST'])
def get_emotion():
    print(f"Received emotion analysis request with text: {request.json.get('text', '')}", flush=True)
    text = request.json.get('text', '')
    emotion = analyze_emotion(text)
    print(f"analyzed response: {emotion}", flush=True)
    return emotion

@app.before_request
def log_request():
    print(f"Incoming {request.method} request to {request.path}", flush=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
