from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(title="Twitter Sentiment Analysis API")

# Add CORS middleware to allow requests from your future frontend
# Allow all origins for simplicity in development and personal projects.
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model and vectorizer
try:
    model = joblib.load('naive_bayes_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
except FileNotFoundError:
    model = None
    vectorizer = None

# Define the request body structure using Pydantic
class Tweet(BaseModel):
    text: str

# Basic text preprocessing function
def preprocess_tweet(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)      # Remove mentions
    text = re.sub(r'#', '', text)         # Remove hashtag symbol
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove punctuation and numbers
    text = text.lower()                   # Convert to lowercase
    text = text.strip()                   # Remove leading/trailing whitespace
    return text

@app.get("/")
def read_root():
    return {"message": "Welcome to the Sentiment Analysis API. Use the /predict endpoint to get predictions."}

@app.post("/predict")
def predict_sentiment(tweet: Tweet):
    if model is None or vectorizer is None:
        return {"error": "Model or vectorizer not found. Make sure 'naive_bayes_model.joblib' and 'tfidf_vectorizer.joblib' are in the same directory."}

    # Preprocess the input tweet
    processed_text = preprocess_tweet(tweet.text)

    # Vectorize the preprocessed text
    vectorized_text = vectorizer.transform([processed_text])

    # Make a prediction
    prediction = model.predict(vectorized_text)
    sentiment_code = int(prediction[0])

    # Map sentiment code to emotion string.
    # Based on testing, the model uses 0 for negative and 1 for positive.
    sentiment_mapping = {0: "negative", 1: "positive"}
    sentiment_emotion = sentiment_mapping.get(sentiment_code, "unknown")

    # Get prediction confidence if available
    try:
        prediction_proba = model.predict_proba(vectorized_text)
        confidence = float(max(prediction_proba[0]))
    except AttributeError:
        confidence = None

    return {
        "original_tweet": tweet.text,
        "processed_text": processed_text,
        "sentiment": sentiment_emotion,
        "confidence": confidence
    }
