"""
Automatic Sample Data Generator
Creates realistic tweet data without API
"""
import pandas as pd
import os
from datetime import datetime, timedelta
import random

class DataGenerator:
    """Generate realistic sample tweets"""
    
    def __init__(self):
        self.sample_tweets = {
            'positive': [
                "I absolutely love Python! Best programming language ever! 🎉",
                "Just finished my first ML project. So proud! #MachineLearning",
                "Python makes data science so easy and fun! 😍",
                "Amazing tutorial on sentiment analysis! This is incredible!",
                "Best programming community! Python developers are so helpful! ❤️",
                "Just got my first data science job! Dreams come true! 🚀",
                "This pandas library is incredible! Saves so much time!",
                "Finally understood neural networks! Feeling accomplished! 💪",
                "Python + AI = Future! So excited!",
                "Great conference today! Learned so much about NLP!",
                "Love how simple Python syntax is! Easy to learn!",
                "Just deployed my first Streamlit app! Awesome! 🎊",
                "Open source community is amazing! Thank you!",
                "Perfect explanation! Mind blown! 🤯",
                "Python is revolutionizing data science! Incredible!",
                "Best decision ever was learning Python! 👍",
                "This project turned out great! Happy!",
                "Such a supportive community! Love it! ❤️",
                "Finally mastered pandas! Feel like a pro!",
                "Excellent documentation! Makes everything easier!",
                "NumPy arrays are so powerful! Game changer!",
                "TensorFlow is amazing for deep learning!",
                "Debugging is easier than other languages!",
                "Jupyter notebooks changed my workflow!",
                "Flask made web development fun!",
            ],
            'negative': [
                "This tutorial is confusing! Worst explanation ever! 😞",
                "Spent 5 hours debugging. Still doesn't work! Frustrated!",
                "This library has too many bugs! Unusable!",
                "Documentation is terrible! No clear examples!",
                "I hate dependency conflicts! Nightmare!",
                "Error makes no sense! Wasted entire day!",
                "Worst course ever! Don't waste money!",
                "Too many breaking changes! Can't keep up!",
                "Performance is terrible! Too slow!",
                "Installation failed again! Ridiculous!",
                "API is so complicated! Why so hard?",
                "Can't believe I paid for this! Waste!",
                "Nothing works as documented! Annoying!",
                "Too many deprecated features! Frustrating!",
                "Error messages are useless! No help!",
                "Framework is overly complicated! Hate it!",
                "Support is non-existent! No one helps!",
                "Version incompatibility nightmare! Give up!",
                "This is broken! How did this pass testing?",
                "Terrible user experience! Disappointed!",
                "Too slow for real-time! Useless!",
                "Garbage collection issues killing performance!",
                "Training takes forever! Awful!",
                "Indentation errors are the worst!",
                "Memory leaks everywhere! Terrible!",
            ],
            'neutral': [
                "Python 3.12 released yesterday. Checking it out.",
                "New version has some changes. Reading docs.",
                "Attended a webinar today. It was okay.",
                "Working on a project this week.",
                "Comparing different libraries for my use case.",
                "Reading about transformers. Interesting.",
                "Downloaded the dataset. Will analyze tomorrow.",
                "Setting up environment. Standard process.",
                "Conference is next month. Planning to attend.",
                "Updated packages today. Everything seems fine.",
                "Testing different models. Collecting results.",
                "Meeting tomorrow. Discussing roadmap.",
                "Reading research papers on NLP.",
                "Installed new IDE. Trying features.",
                "Preprocessing took longer than expected.",
                "Reviewing code from last week. Making notes.",
                "Library updated. Checking changelog.",
                "Working from home today. Usual routine.",
                "Coffee and coding. Regular Monday.",
                "Just debugging code. Nothing special.",
                "Python is a programming language.",
                "NumPy for numerical computing.",
                "Machine learning involves training models.",
                "Pandas provides data structures.",
                "Matplotlib creates visualizations.",
            ]
        }
    
    def generate_tweets(self, num_tweets=100):
        """Generate sample tweets"""
        
        all_tweets = []
        current_time = datetime.now()
        tweet_id = 1000000
        
        # Calculate how many of each sentiment
        tweets_per_sentiment = num_tweets // 3
        
        for sentiment_type, tweets in self.sample_tweets.items():
            for i in range(tweets_per_sentiment):
                # Cycle through available tweets
                text = tweets[i % len(tweets)]
                
                # Realistic metrics based on sentiment
                if sentiment_type == 'positive':
                    likes = random.randint(15, 150)
                    retweets = random.randint(8, 75)
                    replies = random.randint(3, 30)
                elif sentiment_type == 'negative':
                    likes = random.randint(2, 35)
                    retweets = random.randint(0, 15)
                    replies = random.randint(1, 20)
                else:
                    likes = random.randint(5, 45)
                    retweets = random.randint(2, 20)
                    replies = random.randint(1, 12)
                
                all_tweets.append({
                    'id': tweet_id,
                    'text': text,
                    'created_at': current_time - timedelta(hours=random.randint(1, 168)),
                    'likes': likes,
                    'retweets': retweets,
                    'replies': replies,
                    'quotes': random.randint(0, 8)
                })
                tweet_id += 1
        
        # Create DataFrame and shuffle
        df = pd.DataFrame(all_tweets)
        df = df.sample(frac=1).reset_index(drop=True)
        
        return df
    
    def save_data(self, df, filepath='data/raw_tweets.csv'):
        """Save generated data"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        return filepath
    
    def create_sample_data(self, num_tweets=100):
        """Complete workflow: generate and save"""
        print(f"📊 Generating {num_tweets} sample tweets...")
        df = self.generate_tweets(num_tweets)
        filepath = self.save_data(df)
        print(f"✅ Created and saved to {filepath}")
        return df