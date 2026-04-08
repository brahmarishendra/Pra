sentences = [
    "I love this movie",
    "This film is great",
    "Amazing experience",
    "I hate this movie",
    "This film is terrible",
    "Worst experience"
]
labels = [1, 1, 1, 0, 0, 0]

positive_words = ["love", "great", "amazing"]
negative_words = ["hate", "terrible", "worst"]

def predict_sentiment(sentence):
    score = 0
    for w in sentence.lower().split():
        if w in positive_words:
            score += 1
        elif w in negative_words:
            score -= 1
    return 1 if score > 0 else 0

test_sentence = "I love this film"
prediction = predict_sentiment(test_sentence)
print("Sentiment (0=negative, 1=positive):", prediction)

# ---------------- Viva one-line explanation
# Demonstrates sentiment classification using simple word-based scoring logic.
