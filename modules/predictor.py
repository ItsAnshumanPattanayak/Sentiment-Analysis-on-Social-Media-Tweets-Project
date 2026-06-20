"""
Sentiment Prediction Module
Makes predictions on new text using trained model
"""

import joblib
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings

warnings.filterwarnings('ignore')


class SentimentPredictor:
    """
    Sentiment Prediction using trained model
    """
    
    def __init__(self, model_dir='models'):
        """
        Initialize predictor by loading saved models
        
        Args:
            model_dir (str): Directory containing saved models
        """
        try:
            # Load model components
            self.model = joblib.load(f'{model_dir}/sentiment_model.pkl')
            self.vectorizer = joblib.load(f'{model_dir}/vectorizer.pkl')
            self.label_encoder = joblib.load(f'{model_dir}/label_encoder.pkl')
            
            # Initialize preprocessing tools
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            
            # Additional custom stopwords
            custom_stopwords = {'rt', 'http', 'https', 'amp'}
            self.stop_words.update(custom_stopwords)
            
            print("✅ Sentiment Predictor loaded successfully!")
            print(f"   Model type: {type(self.model).__name__}")
            print(f"   Classes: {self.label_encoder.classes_}")
            
        except FileNotFoundError as e:
            print(f"❌ Model files not found in '{model_dir}/'")
            print("Please train the model first!")
            raise
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def clean_text(self, text):
        """
        Clean text (same as preprocessing)
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text) or not text:
            return ""
        
        text = str(text)
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags (keep text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def preprocess(self, text):
        """
        Full preprocessing pipeline
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        # Clean text
        text = self.clean_text(text)
        
        if not text:
            return ""
        
        # Tokenize
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        # Remove stopwords and short words
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        return ' '.join(tokens)
    
    def predict(self, text):
        """
        Predict sentiment of text
        
        Args:
            text (str): Input text
            
        Returns:
            tuple: (sentiment, confidence, probability_dict)
        """
        # Preprocess text
        cleaned_text = self.preprocess(text)
        
        if not cleaned_text:
            return 'Neutral', 0.0, {'Positive': 0.0, 'Negative': 0.0, 'Neutral': 100.0}
        
        # Vectorize
        text_vectorized = self.vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = self.model.predict(text_vectorized)[0]
        sentiment = self.label_encoder.inverse_transform([prediction])[0]
        
        # Get probabilities if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(text_vectorized)[0]
            confidence = max(probabilities) * 100
            
            # Create probability dictionary
            proba_dict = {
                label: prob * 100
                for label, prob in zip(self.label_encoder.classes_, probabilities)
            }
        elif hasattr(self.model, 'decision_function'):
            # For SVM
            decision = self.model.decision_function(text_vectorized)[0]
            
            if isinstance(decision, np.ndarray):
                # Multi-class
                from scipy.special import softmax
                probabilities = softmax(decision)
                confidence = max(probabilities) * 100
                
                proba_dict = {
                    label: prob * 100
                    for label, prob in zip(self.label_encoder.classes_, probabilities)
                }
            else:
                # Binary
                confidence = abs(decision) * 10  # Approximate confidence
                confidence = min(confidence, 100)
                
                proba_dict = {sentiment: confidence}
        else:
            # Default confidence
            confidence = 100.0
            proba_dict = {sentiment: 100.0}
        
        return sentiment, confidence, proba_dict
    
    def predict_batch(self, texts):
        """
        Predict sentiment for multiple texts
        
        Args:
            texts (list): List of texts
            
        Returns:
            pd.DataFrame: Predictions dataframe
        """
        results = []
        
        for text in texts:
            sentiment, confidence, proba_dict = self.predict(text)
            
            result = {
                'text': text,
                'sentiment': sentiment,
                'confidence': confidence
            }
            result.update(proba_dict)
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def analyze_text(self, text, show_details=True):
        """
        Analyze text and display detailed results
        
        Args:
            text (str): Input text
            show_details (bool): Show preprocessing details
        """
        print("="*60)
        print("SENTIMENT ANALYSIS")
        print("="*60)
        
        print(f"\n📝 Original Text:")
        print(f"   {text}")
        
        if show_details:
            cleaned = self.clean_text(text)
            preprocessed = self.preprocess(text)
            
            print(f"\n🧹 Cleaned Text:")
            print(f"   {cleaned}")
            
            print(f"\n🔧 Preprocessed Text:")
            print(f"   {preprocessed}")
        
        # Predict
        sentiment, confidence, proba_dict = self.predict(text)
        
        # Emoji mapping
        emoji_map = {
            'Positive': '😊',
            'Negative': '😞',
            'Neutral': '😐'
        }
        
        print(f"\n{emoji_map.get(sentiment, '📊')} Sentiment: {sentiment}")
        print(f"🎯 Confidence: {confidence:.2f}%")
        
        print(f"\n📊 Probability Distribution:")
        for label, prob in sorted(proba_dict.items(), key=lambda x: x[1], reverse=True):
            bar_length = int(prob / 2)  # Scale to 50 chars max
            bar = '█' * bar_length
            print(f"   {emoji_map.get(label, '📌')} {label:10s}: {bar} {prob:.2f}%")
        
        print("="*60)
        
        return sentiment, confidence, proba_dict


# Test function
if __name__ == "__main__":
    import numpy as np
    
    print("="*60)
    print("SENTIMENT PREDICTION TEST")
    print("="*60)
    
    # Initialize predictor
    try:
        predictor = SentimentPredictor()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease train the model first by running:")
        print("  python modules/model_trainer.py")
        exit()
    
    # Test examples
    test_examples = [
        "I absolutely love this product! It's amazing and exceeded all my expectations! 🎉",
        "This is the worst experience ever. Terrible service and quality. Very disappointed.",
        "It's okay. Nothing special, but not bad either. Just average.",
        "Python is an incredible programming language! Great for data science.",
        "I hate waiting in long queues. So frustrating and annoying!",
        "The weather is nice today. Going for a walk in the park.",
        "Best movie I've ever seen! The plot was brilliant! 10/10",
        "Don't waste your money on this. Complete garbage.",
        "Meh, could be better, could be worse.",
        "Fantastic! Absolutely brilliant work! Keep it up! 👏"
    ]
    
    print("\n🧪 Testing with sample texts:\n")
    
    results = []
    
    for i, text in enumerate(test_examples, 1):
        print(f"\n{'='*60}")
        print(f"Example {i}/{len(test_examples)}")
        print('='*60)
        
        sentiment, confidence, proba_dict = predictor.analyze_text(text, show_details=False)
        
        results.append({
            'text': text[:50] + '...' if len(text) > 50 else text,
            'sentiment': sentiment,
            'confidence': f"{confidence:.1f}%"
        })
    
    # Summary
    print("\n\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Interactive mode
    print("\n\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("\n💡 Enter your own text to analyze (or 'quit' to exit)")
    
    while True:
        print("\n" + "-"*60)
        user_input = input("\nEnter text: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if user_input:
            predictor.analyze_text(user_input)
        else:
            print("⚠️ Please enter some text!")