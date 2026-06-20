"""
Tweet Collection Module
Collects tweets using Twitter API v2
"""

import tweepy
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime
import time

# Load environment variables
load_dotenv()


class TweetCollector:
    """
    Collects tweets from Twitter API v2
    """
    
    def __init__(self):
        """Initialize Twitter API Client"""
        try:
            self.client = tweepy.Client(
                bearer_token=os.getenv('BEARER_TOKEN'),
                consumer_key=os.getenv('API_KEY'),
                consumer_secret=os.getenv('API_SECRET'),
                access_token=os.getenv('ACCESS_TOKEN'),
                access_token_secret=os.getenv('ACCESS_TOKEN_SECRET'),
                wait_on_rate_limit=True
            )
            print("✅ Twitter API Client initialized successfully!")
        except Exception as e:
            print(f"❌ Error initializing Twitter API: {e}")
            raise
    
    def collect_tweets(self, query, max_tweets=500):
        """
        Collect tweets based on search query
        
        Args:
            query (str): Search query
            max_tweets (int): Maximum number of tweets to collect
            
        Returns:
            pd.DataFrame: DataFrame containing collected tweets
        """
        tweets_data = []
        
        try:
            print(f"🔍 Searching for tweets about '{query}'...")
            
            # Search recent tweets
            for tweet in tweepy.Paginator(
                self.client.search_recent_tweets,
                query=f"{query} -is:retweet lang:en",
                tweet_fields=['created_at', 'public_metrics', 'lang', 'author_id'],
                max_results=100
            ).flatten(limit=max_tweets):
                
                tweets_data.append({
                    'id': tweet.id,
                    'text': tweet.text,
                    'created_at': tweet.created_at,
                    'likes': tweet.public_metrics['like_count'],
                    'retweets': tweet.public_metrics['retweet_count'],
                    'replies': tweet.public_metrics['reply_count'],
                    'quotes': tweet.public_metrics.get('quote_count', 0)
                })
                
                # Progress update
                if len(tweets_data) % 50 == 0:
                    print(f"📊 Collected {len(tweets_data)} tweets...")
            
            print(f"\n✅ Total tweets collected: {len(tweets_data)}")
            
            # Create DataFrame
            df = pd.DataFrame(tweets_data)
            
            # Save to CSV
            os.makedirs('data', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'data/raw_tweets_{timestamp}.csv'
            df.to_csv(filename, index=False)
            df.to_csv('data/raw_tweets.csv', index=False)  # Also save without timestamp
            
            print(f"💾 Saved to {filename}")
            
            return df
            
        except tweepy.errors.TweepyException as e:
            print(f"❌ Twitter API Error: {e}")
            return None
        except Exception as e:
            print(f"❌ Error collecting tweets: {e}")
            return None
    
    def collect_user_tweets(self, username, max_tweets=100):
        """
        Collect tweets from a specific user
        
        Args:
            username (str): Twitter username (without @)
            max_tweets (int): Maximum number of tweets
            
        Returns:
            pd.DataFrame: User's tweets
        """
        try:
            user = self.client.get_user(username=username)
            
            if not user.data:
                print(f"❌ User @{username} not found")
                return None
            
            tweets = self.client.get_users_tweets(
                id=user.data.id,
                max_results=max_tweets,
                tweet_fields=['created_at', 'public_metrics']
            )
            
            tweets_data = []
            for tweet in tweets.data:
                tweets_data.append({
                    'id': tweet.id,
                    'text': tweet.text,
                    'created_at': tweet.created_at,
                    'likes': tweet.public_metrics['like_count'],
                    'retweets': tweet.public_metrics['retweet_count'],
                    'replies': tweet.public_metrics['reply_count']
                })
            
            df = pd.DataFrame(tweets_data)
            print(f"✅ Collected {len(df)} tweets from @{username}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def get_trending_topics(self, woeid=1):
        """
        Get trending topics (requires elevated access)
        
        Args:
            woeid (int): Where On Earth ID (1 = Worldwide)
            
        Returns:
            list: Trending topics
        """
        try:
            # Note: This requires API v1.1 and elevated access
            auth = tweepy.OAuth1UserHandler(
                os.getenv('API_KEY'),
                os.getenv('API_SECRET'),
                os.getenv('ACCESS_TOKEN'),
                os.getenv('ACCESS_TOKEN_SECRET')
            )
            api = tweepy.API(auth)
            
            trends = api.get_place_trends(woeid)
            trending_topics = [trend['name'] for trend in trends[0]['trends'][:10]]
            
            return trending_topics
            
        except Exception as e:
            print(f"❌ Error getting trends: {e}")
            return []


# Test function
if __name__ == "__main__":
    collector = TweetCollector()
    
    # Test collection
    query = input("Enter search query (default: Python): ").strip() or "Python"
    max_tweets = int(input("Number of tweets (default: 100): ").strip() or "100")
    
    df = collector.collect_tweets(query, max_tweets)
    
    if df is not None:
        print("\n📊 Sample tweets:")
        print(df.head())
        print("\n📈 Statistics:")
        print(df.describe())