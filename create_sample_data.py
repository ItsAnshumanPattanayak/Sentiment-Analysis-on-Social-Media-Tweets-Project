"""
Script to create sample tweets data for testing without Twitter API
"""
import pandas as pd
import os
from datetime import datetime, timedelta
import random

# Sample tweets
sample_tweets_positive = [
    "I love Python! It's the best programming language for data science.",
    "AI technology is amazing and changing the world for the better!",
    "This movie was incredible! Best film I've seen all year.",
    "I'm so happy with this product. Highly recommend!",
    "Climate action is crucial and inspiring! Great initiative.",
    "Had an amazing time at the conference. Learned so much!",
    "Great customer service! They solved my problem instantly.",
    "This book is fantastic! Couldn't put it down.",
    "Loving the new features in the latest update!",
    "Best vacation ever! #TravelBlog"
]

sample_tweets_negative = [
    "Python can be frustrating sometimes with all the syntax errors.",
    "AI bias is a serious problem that needs addressing.",
    "Terrible movie. Complete waste of time and money.",
    "Very disappointed with the product quality. Not worth it.",
    "Climate change is terrifying and we're running out of time.",
    "The conference was boring and poorly organized.",
    "Awful customer service experience. Won't be back.",
    "Could not finish this book. Absolutely dreadful.",
    "The new update ruined everything. Bring back the old version!",
    "Worst vacation disaster ever! #NeverAgain"
]

sample_tweets_neutral = [
    "Python is a programming language used for many purposes.",
    "AI continues to develop in various fields.",
    "The movie has good acting and okay plot.",
    "This product does what it claims to do.",
    "Climate data shows various trends over time.",
    "The conference had several interesting talks.",
    "The customer service was available during business hours.",
    "This book has 300 pages.",
    "The latest version has new features.",
    "Traveled to several countries this year."
]

def create_sample_data(num_tweets=500):
    """Create sample dataset"""
    tweets = []
    
    # Create a mix of positive, negative, and neutral tweets
    positive_count = int(num_tweets * 0.4)
    negative_count = int(num_tweets * 0.35)
    neutral_count = num_tweets - positive_count - negative_count
    
    # Add positive tweets
    for i in range(positive_count):
        tweets.append({
            'id': 1000000 + i,
            'text': random.choice(sample_tweets_positive),
            'created_at': datetime.now() - timedelta(days=random.randint(1, 30)),
            'likes': random.randint(10, 1000),
            'retweets': random.randint(5, 500),
            'replies': random.randint(1, 100),
            'quotes': random.randint(0, 50)
        })
    
    # Add negative tweets
    for i in range(negative_count):
        tweets.append({
            'id': 2000000 + i,
            'text': random.choice(sample_tweets_negative),
            'created_at': datetime.now() - timedelta(days=random.randint(1, 30)),
            'likes': random.randint(10, 1000),
            'retweets': random.randint(5, 500),
            'replies': random.randint(1, 100),
            'quotes': random.randint(0, 50)
        })
    
    # Add neutral tweets
    for i in range(neutral_count):
        tweets.append({
            'id': 3000000 + i,
            'text': random.choice(sample_tweets_neutral),
            'created_at': datetime.now() - timedelta(days=random.randint(1, 30)),
            'likes': random.randint(10, 1000),
            'retweets': random.randint(5, 500),
            'replies': random.randint(1, 100),
            'quotes': random.randint(0, 50)
        })
    
    # Create DataFrame
    df = pd.DataFrame(tweets)
    
    # Save to CSV
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/raw_tweets.csv', index=False)
    
    print(f"✅ Created {len(df)} sample tweets in data/raw_tweets.csv")
    print(f"   - {positive_count} positive tweets")
    print(f"   - {negative_count} negative tweets")
    print(f"   - {neutral_count} neutral tweets")

if __name__ == "__main__":
    create_sample_data(500)
    print("\n💡 Now run: streamlit run app.py")
    print("   Go to '🧹 Preprocess' to start the pipeline!")
