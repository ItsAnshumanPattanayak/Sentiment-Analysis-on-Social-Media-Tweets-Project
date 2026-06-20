"""
Text Preprocessing Module
Cleans and preprocesses tweets for sentiment analysis
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)


class TextPreprocessor:
    """
    Text preprocessing for sentiment analysis
    """
    
    def __init__(self):
        """Initialize preprocessor with stopwords and lemmatizer"""
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Additional custom stopwords
        custom_stopwords = {'rt', 'http', 'https', 'amp'}
        self.stop_words.update(custom_stopwords)
        
        print("✅ Text Preprocessor initialized")
    
    def clean_text(self, text):
        """
        Clean and normalize text
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags (keep the text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def remove_stopwords(self, text):
        """
        Remove stopwords from text
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text without stopwords
        """
        if not text:
            return ""
        
        try:
            tokens = word_tokenize(text)
            filtered_tokens = [word for word in tokens if word.lower() not in self.stop_words and len(word) > 2]
            return ' '.join(filtered_tokens)
        except Exception as e:
            print(f"Error removing stopwords: {e}")
            return text
    
    def lemmatize_text(self, text):
        """
        Lemmatize text
        
        Args:
            text (str): Input text
            
        Returns:
            str: Lemmatized text
        """
        if not text:
            return ""
        
        try:
            tokens = word_tokenize(text)
            lemmatized = [self.lemmatizer.lemmatize(word) for word in tokens]
            return ' '.join(lemmatized)
        except Exception as e:
            print(f"Error lemmatizing: {e}")
            return text
    
    def get_sentiment_label(self, text):
        """
        Get sentiment label using TextBlob
        
        Args:
            text (str): Input text
            
        Returns:
            str: Sentiment label (Positive, Negative, Neutral)
        """
        if not text or pd.isna(text):
            return 'Neutral'
        
        try:
            blob = TextBlob(str(text))
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                return 'Positive'
            elif polarity < -0.1:
                return 'Negative'
            else:
                return 'Neutral'
        except Exception as e:
            print(f"Error getting sentiment: {e}")
            return 'Neutral'
    
    def get_sentiment_score(self, text):
        """
        Get detailed sentiment scores
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Sentiment scores (polarity, subjectivity)
        """
        if not text or pd.isna(text):
            return {'polarity': 0.0, 'subjectivity': 0.0}
        
        try:
            blob = TextBlob(str(text))
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except Exception as e:
            print(f"Error getting sentiment scores: {e}")
            return {'polarity': 0.0, 'subjectivity': 0.0}
    
    def preprocess_dataframe(self, df, text_column='text'):
        """
        Preprocess entire dataframe
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Name of text column
            
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        print("🧹 Starting preprocessing...")
        
        if text_column not in df.columns:
            print(f"❌ Column '{text_column}' not found!")
            return df
        
        # Create a copy
        df = df.copy()
        
        # Clean text
        print("1️⃣ Cleaning text...")
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        
        # Remove stopwords
        print("2️⃣ Removing stopwords...")
        df['cleaned_text'] = df['cleaned_text'].apply(self.remove_stopwords)
        
        # Lemmatize
        print("3️⃣ Lemmatizing text...")
        df['cleaned_text'] = df['cleaned_text'].apply(self.lemmatize_text)
        
        # Get sentiment
        print("4️⃣ Analyzing sentiment...")
        df['sentiment'] = df[text_column].apply(self.get_sentiment_label)
        
        # Get sentiment scores
        sentiment_scores = df[text_column].apply(self.get_sentiment_score)
        df['polarity'] = sentiment_scores.apply(lambda x: x['polarity'])
        df['subjectivity'] = sentiment_scores.apply(lambda x: x['subjectivity'])
        
        # Remove empty texts
        df = df[df['cleaned_text'].str.strip() != '']
        
        # Save processed data
        import os
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/processed_tweets.csv', index=False)
        
        print(f"\n✅ Preprocessing completed! {len(df)} tweets processed")
        print(f"💾 Saved to data/processed_tweets.csv")
        
        return df
    
    def get_word_frequency(self, df, sentiment=None, top_n=20):
        """
        Get word frequency for sentiment analysis
        
        Args:
            df (pd.DataFrame): Preprocessed dataframe
            sentiment (str): Filter by sentiment (optional)
            top_n (int): Number of top words
            
        Returns:
            dict: Word frequency
        """
        from collections import Counter
        
        if sentiment:
            texts = df[df['sentiment'] == sentiment]['cleaned_text']
        else:
            texts = df['cleaned_text']
        
        # Combine all texts
        all_words = ' '.join(texts).split()
        
        # Count frequency
        word_freq = Counter(all_words).most_common(top_n)
        
        return dict(word_freq)


# Test function
if __name__ == "__main__":
    import os
    
    # Test preprocessing
    preprocessor = TextPreprocessor()
    
    # Sample texts
    sample_texts = [
        "I love this product! It's amazing! 😊 #awesome",
        "This is the worst thing ever. I hate it! 😞",
        "It's okay, nothing special. Just average.",
        "@user Check out this link: https://example.com #Python",
        "RT @someone: Great job! Keep it up! 👍"
    ]
    
    print("🧪 Testing Text Preprocessing\n")
    print("="*60)
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\n📝 Original {i}: {text}")
        
        cleaned = preprocessor.clean_text(text)
        print(f"🧹 Cleaned: {cleaned}")
        
        no_stop = preprocessor.remove_stopwords(cleaned)
        print(f"🗑️ No Stopwords: {no_stop}")
        
        lemmatized = preprocessor.lemmatize_text(no_stop)
        print(f"📖 Lemmatized: {lemmatized}")
        
        sentiment = preprocessor.get_sentiment_label(text)
        print(f"😊 Sentiment: {sentiment}")
        
        print("-"*60)
    
    # Test with CSV if available
    if os.path.exists('data/raw_tweets.csv'):
        print("\n\n📊 Processing CSV file...")
        df = pd.read_csv('data/raw_tweets.csv')
        df_processed = preprocessor.preprocess_dataframe(df)
        
        print("\n📈 Sentiment Distribution:")
        print(df_processed['sentiment'].value_counts())
        
        print("\n🔤 Top 10 words:")
        top_words = preprocessor.get_word_frequency(df_processed, top_n=10)
        for word, count in top_words.items():
            print(f"  {word}: {count}")