"""
Automatic Setup Helper
Ensures project is ready on first run
"""
import os
import sys
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.data_generator import DataGenerator
from modules.preprocessor import TextPreprocessor
from modules.model_trainer import SentimentClassifier

class SetupHelper:
    """Auto-setup project on first run"""
    
    def __init__(self):
        self.project_ready = False
    
    def check_data(self):
        """Check if data exists"""
        return os.path.exists('data/raw_tweets.csv')
    
    def check_processed_data(self):
        """Check if processed data exists"""
        return os.path.exists('data/processed_tweets.csv')
    
    def check_model(self):
        """Check if model is trained"""
        return (os.path.exists('models/sentiment_model.pkl') and
                os.path.exists('models/vectorizer.pkl') and
                os.path.exists('models/label_encoder.pkl'))
    
    def create_data(self):
        """Create sample data"""
        print("📊 Creating sample data...")
        generator = DataGenerator()
        df = generator.create_sample_data(num_tweets=100)
        print(f"✅ Generated {len(df)} tweets")
        return df
    
    def preprocess_data(self):
        """Preprocess data"""
        print("🧹 Preprocessing data...")
        df = pd.read_csv('data/raw_tweets.csv')
        preprocessor = TextPreprocessor()
        df_processed = preprocessor.preprocess_dataframe(df)
        print(f"✅ Preprocessed {len(df_processed)} tweets")
        return df_processed
    
    def train_model(self):
        """Train model"""
        print("🤖 Training model...")
        df = pd.read_csv('data/processed_tweets.csv')
        
        classifier = SentimentClassifier(max_features=5000)
        X_train, X_test, y_train, y_test = classifier.prepare_data(df)
        
        # Train with Logistic Regression (best performance)
        classifier.train_model(X_train, y_train, 'logistic')
        accuracy, _ = classifier.evaluate_model(X_test, y_test)
        
        classifier.save_model()
        print(f"✅ Model trained with {accuracy:.2%} accuracy")
        return accuracy
    
    def setup_project(self, force=False):
        """Complete auto-setup"""
        print("\n" + "="*60)
        print("🚀 AUTOMATIC PROJECT SETUP")
        print("="*60 + "\n")
        
        steps_completed = []
        
        # Step 1: Check/Create data
        if not self.check_data() or force:
            print("Step 1/3: Creating sample data...")
            self.create_data()
            steps_completed.append("Data created")
        else:
            print("Step 1/3: ✅ Data already exists")
        
        # Step 2: Check/Preprocess
        if not self.check_processed_data() or force:
            print("\nStep 2/3: Preprocessing data...")
            self.preprocess_data()
            steps_completed.append("Data preprocessed")
        else:
            print("Step 2/3: ✅ Data already preprocessed")
        
        # Step 3: Check/Train model
        if not self.check_model() or force:
            print("\nStep 3/3: Training model...")
            self.train_model()
            steps_completed.append("Model trained")
        else:
            print("Step 3/3: ✅ Model already trained")
        
        print("\n" + "="*60)
        if steps_completed:
            print("✅ SETUP COMPLETE!")
            print(f"   Completed: {', '.join(steps_completed)}")
        else:
            print("✅ PROJECT ALREADY READY!")
        print("="*60 + "\n")
        
        self.project_ready = True
        return True
    
    def get_status(self):
        """Get current project status"""
        return {
            'has_data': self.check_data(),
            'has_processed_data': self.check_processed_data(),
            'has_model': self.check_model(),
            'ready': self.check_data() and self.check_processed_data() and self.check_model()
        }