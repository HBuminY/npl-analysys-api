from flask import Flask, request, jsonify
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download required NLTK data
nltk.download('vader_lexicon', quiet=True)

app = Flask(__name__)

def analyze_emotion(text):
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    pos = scores['pos']
    neg = scores['neg']
    neu = scores['neu']

    text_lower = text.lower()

    emotional_avg = {"happy": 0, "sad": 0, "angry": 0, "surprised": 0}

    # Keyword-based hints
    happy_keywords = ["love", "great", "awesome", "fantastic", "joy", "glad", "good"]
    sad_keywords = ["sad", "down", "depressed", "upset", "cry", "unhappy"]
    angry_keywords = ["angry", "mad", "furious", "annoyed", "hate", "irritated"]
    surprised_keywords = ["wow", "surprised", "surprise", "unexpected", "shocked", "amazed", "astonished"]

    # Check keywords first
    for word in happy_keywords:
        if word in text_lower:
            emotional_avg["happy"] += 0.5
    for word in sad_keywords:
        if word in text_lower:
            emotional_avg["sad"] += 0.5
    for word in angry_keywords:
        if word in text_lower:
            emotional_avg["angry"] += 0.5
    for word in surprised_keywords:
        if word in text_lower:
            emotional_avg["surprised"] += 0.5

    # If no keyword found, fallback to sentiment
    if compound >= 0.5 and pos > 0.5:
        emotional_avg["happy"] += 0.4
    elif compound <= -0.5 and neg > 0.5:
        emotional_avg["angry"] += 0.4
    elif -0.3 < compound < 0.3 and neu > 0.6:
        emotional_avg["surprised"] += 0.4
    else:
        emotional_avg["sad"] += 0.4

    result = max(emotional_avg, key=emotional_avg.get)
    return result

@app.route('/')
def hello():
    return "Hello World!"

@app.route('/emotion', methods=['POST'])
def get_emotion():
    print(f"Received emotion analysis request with text: {request.json.get('text', '')}", flush=True)
    text = request.json.get('text', '')
    emotion = analyze_emotion(text)
    print(f"Analyzed response: {emotion}", flush=True)
    return jsonify({"emotion": emotion.emotion})

@app.before_request
def log_request():
    print(f"Incoming {request.method} request to {request.path}", flush=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
