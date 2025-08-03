import tweepy
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
import os

# ==== Twitter API v2 Bearer Token ====
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAP%2FH3AEAAAAABR3KsWBl1CKvsgfIeuFp8jmM91M%3DKknFESaAmhrcAbqy5imlCvYCEu5nz8O5ZylZtxzc0uUoiKgi2B"

# ==== Authenticate with Twitter API v2 ====
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# ==== Analyze Sentiment ====
def analyze_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# ==== Fetch and Analyze Tweets ====
def fetch_tweets_v2(keyword, max_tweets=100):
    print(f"Fetching tweets for: {keyword}")
    tweets_data = []

    query = f"{keyword} -is:retweet lang:en"

    try:
        tweets = client.search_recent_tweets(query=query, max_results=100, tweet_fields=['created_at'])

        if tweets.data:
            for tweet in tweets.data:
                text = tweet.text
                sentiment = analyze_sentiment(text)
                created_at = tweet.created_at
                tweets_data.append({
                    'Time': created_at,
                    'Hour': created_at.hour,
                    'Minute': created_at.minute,
                    'Second': created_at.second,
                    'Day': created_at.day,
                    'Weekday': created_at.weekday(),
                    'Month': created_at.month,
                    'Year': created_at.year,
                    'Tweet': text,
                    'Sentiment': sentiment
                })
        else:
            print("No tweets found.")
    except Exception as e:
        print("Failed to fetch tweets:", e)

    return tweets_data

# ==== Train AI Model ====
def train_time_sentiment_model(df):
    features = ['Hour', 'Minute', 'Second', 'Day', 'Weekday', 'Month', 'Year']
    X = df[features]
    y = df['Sentiment']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_encoded)

    # Save model and encoder
    with open("sentiment_time_model.pkl", "wb") as f:
        pickle.dump((model, le), f)

    print("AI model trained and saved.")

# ==== Analyze Sentiment Peaks ====
def analyze_sentiment_peaks(df, sentiment):
    print(f"\nPeak time analysis for '{sentiment}' sentiment:")
    filtered = df[df['Sentiment'] == sentiment]

    if filtered.empty:
        print("No data for this sentiment.")
        return

    for col in ['Hour', 'Minute', 'Second', 'Weekday', 'Month', 'Year']:
        common = Counter(filtered[col]).most_common(3)
        print(f"Most common {col}s: {[f'{val} ({count})' for val, count in common]}")

# ==== Predict Future Sentiment ====
def predict_future_sentiment(date_str):
    try:
        with open("sentiment_time_model.pkl", "rb") as f:
            model, le = pickle.load(f)
    except FileNotFoundError:
        print("Model not found. Train it first using tweet data or CSV.")
        return

    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M")
    except ValueError:
        print("Invalid date format. Use 'YYYY-MM-DD HH:MM'")
        return

    features = [[
        dt.hour,
        dt.minute,
        dt.second,
        dt.day,
        dt.weekday(),
        dt.month,
        dt.year
    ]]

    pred_encoded = model.predict(features)[0]
    pred_label = le.inverse_transform([pred_encoded])[0]

    print(f"\nPrediction for {date_str}:")
    print(f"Likely sentiment: **{pred_label.upper()}**")

# ==== Save, Plot, Train ====
def save_and_plot(tweets_data, keyword):
    df = pd.DataFrame(tweets_data)
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"tweets_{keyword}_{now}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} tweets to {filename}")

    sentiment_counts = df['Sentiment'].value_counts()
    sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'])
    plt.title(f"Sentiment Analysis for '{keyword}'")
    plt.xlabel("Sentiment")
    plt.ylabel("Tweet Count")
    plt.grid(axis='y')
    plt.show()

    train_time_sentiment_model(df)

    for sentiment in ['Positive', 'Negative', 'Neutral']:
        analyze_sentiment_peaks(df, sentiment)

# ==== Main ====
if __name__ == "__main__":
    keyword = input("Enter keyword to search: ").strip()

    tweets = fetch_tweets_v2(keyword)

    if tweets:
        save_and_plot(tweets, keyword)
    else:
        print("\nFalling back to CSV file.")
        csv_path = input("Enter path to CSV file (e.g., tweets_elon_2025.csv): ").strip()

        if not os.path.exists(csv_path):
            print("File not found. Exiting.")
        else:
            df = pd.read_csv(csv_path)
            print(f"Loaded {len(df)} rows from CSV.")

            train_time_sentiment_model(df)

            for sentiment in ['Positive', 'Negative', 'Neutral']:
                analyze_sentiment_peaks(df, sentiment)

    # Predict future sentiment
    ask = input("\nWant to predict sentiment for future date/time? (y/n): ").lower()
    if ask == 'y':
        future_time = input("Enter future datetime (YYYY-MM-DD HH:MM): ")
        predict_future_sentiment(future_time)
