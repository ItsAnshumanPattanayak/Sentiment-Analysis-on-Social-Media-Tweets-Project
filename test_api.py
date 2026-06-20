"""
Test Twitter API Connection
"""
import tweepy
import os
from dotenv import load_dotenv

load_dotenv()

print("="*60)
print("TESTING TWITTER API CONNECTION")
print("="*60)

# Load credentials
api_key = os.getenv('API_KEY')
api_secret = os.getenv('API_SECRET')
bearer_token = os.getenv('BEARER_TOKEN')
access_token = os.getenv('ACCESS_TOKEN')
access_token_secret = os.getenv('ACCESS_TOKEN_SECRET')

# Check if credentials exist
print("\n1️⃣ Checking credentials...")
if not all([api_key, api_secret, bearer_token, access_token, access_token_secret]):
    print("❌ Missing credentials in .env file!")
    print("\nMake sure your .env has:")
    print("API_KEY=...")
    print("API_SECRET=...")
    print("BEARER_TOKEN=...")
    print("ACCESS_TOKEN=...")
    print("ACCESS_TOKEN_SECRET=...")
    exit()
else:
    print("✅ All credentials found in .env")

# Test connection
print("\n2️⃣ Testing API connection...")
try:
    client = tweepy.Client(
        bearer_token=bearer_token,
        consumer_key=api_key,
        consumer_secret=api_secret,
        access_token=access_token,
        access_token_secret=access_token_secret
    )
    
    # Test 1: Get authenticated user
    print("\n3️⃣ Testing authentication...")
    me = client.get_me()
    print(f"✅ Authenticated as: @{me.data.username}")
    
    # Test 2: Search tweets
    print("\n4️⃣ Testing tweet search...")
    response = client.search_recent_tweets(
        query="python -is:retweet lang:en",
        max_results=10,
        tweet_fields=['created_at', 'public_metrics']
    )
    
    if response.data:
        print(f"✅ Found {len(response.data)} tweets!")
        print("\nSample tweet:")
        print(f"  {response.data[0].text[:100]}...")
    else:
        print("⚠️ No tweets found, but API is working!")
        print("This might be due to:")
        print("  - Rate limits")
        print("  - API access level restrictions")
        print("  - Query matching no recent tweets")
    
    print("\n" + "="*60)
    print("✅ API CONNECTION SUCCESSFUL!")
    print("="*60)
    
except tweepy.errors.Unauthorized as e:
    print("\n❌ AUTHENTICATION ERROR!")
    print("Your API credentials are invalid.")
    print("\nSolutions:")
    print("1. Check if credentials are correct in .env")
    print("2. Regenerate keys in Twitter Developer Portal")
    print("3. Make sure app has Read permissions")
    print(f"\nError details: {e}")
    
except tweepy.errors.Forbidden as e:
    print("\n❌ FORBIDDEN ERROR!")
    print("You don't have access to this endpoint.")
    print("\nThis usually means:")
    print("1. Your Twitter API access level is too low (Free tier)")
    print("2. You need Essential/Elevated/Premium access")
    print("3. Apply for elevated access at: https://developer.twitter.com/")
    print(f"\nError details: {e}")
    
except tweepy.errors.TooManyRequests as e:
    print("\n❌ RATE LIMIT ERROR!")
    print("You've made too many requests.")
    print("Wait 15 minutes and try again.")
    print(f"\nError details: {e}")
    
except Exception as e:
    print(f"\n❌ UNEXPECTED ERROR: {e}")
    print("\nCheck your internet connection and try again.")