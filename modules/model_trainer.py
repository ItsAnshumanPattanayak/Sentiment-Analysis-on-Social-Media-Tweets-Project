"""
Model Training Module
Trains sentiment classification models using ML algorithms
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support
)
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


class SentimentClassifier:
    """
    Sentiment Classification Model Trainer
    """
    
    def __init__(self, max_features=5000):
        """
        Initialize classifier
        
        Args:
            max_features (int): Maximum number of features for TF-IDF
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        self.model = None
        self.label_encoder = LabelEncoder()
        self.max_features = max_features
        
        print(f"✅ Sentiment Classifier initialized (max_features={max_features})")
    
    def prepare_data(self, df, text_column='cleaned_text', label_column='sentiment', test_size=0.2):
        """
        Prepare data for training
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Name of text column
            label_column (str): Name of label column
            test_size (float): Test set size
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        print("📊 Preparing data for training...")
        
        # Encode labels
        y = self.label_encoder.fit_transform(df[label_column])
        
        # Vectorize text using TF-IDF
        print(f"🔢 Vectorizing text (TF-IDF, max_features={self.max_features})...")
        X = self.vectorizer.fit_transform(df[text_column])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )
        
        print(f"✅ Data prepared:")
        print(f"   Training samples: {X_train.shape[0]}")
        print(f"   Testing samples: {X_test.shape[0]}")
        print(f"   Features: {X_train.shape[1]}")
        print(f"   Classes: {self.label_encoder.classes_}")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train, model_type='logistic'):
        """
        Train sentiment classification model
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_type (str): Type of model ('logistic', 'naive_bayes', 'svm', 'random_forest')
        """
        print(f"\n🤖 Training {model_type} model...")
        
        if model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced',
                solver='lbfgs'
            )
        elif model_type == 'naive_bayes':
            self.model = MultinomialNB(alpha=1.0)
        elif model_type == 'svm':
            self.model = LinearSVC(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            )
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        self.model.fit(X_train, y_train)
        
        print(f"✅ Model trained successfully!")
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            tuple: accuracy, classification_report
        """
        print("\n📈 Evaluating model...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Classification report
        report = classification_report(
            y_test,
            y_pred,
            target_names=self.label_encoder.classes_,
            digits=4
        )
        
        # Precision, Recall, F1-Score
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test,
            y_pred,
            average='weighted'
        )
        
        print(f"\n🎯 Model Performance:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        
        print(f"\n📊 Detailed Classification Report:")
        print("="*60)
        print(report)
        
        return accuracy, report
    
    def plot_confusion_matrix(self, X_test, y_test, save_path='visualizations/confusion_matrix.png'):
        """
        Plot confusion matrix
        
        Args:
            X_test: Test features
            y_test: Test labels
            save_path (str): Path to save plot
        """
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        # Save plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n💾 Confusion matrix saved to {save_path}")
        plt.close()
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation
        
        Args:
            X: Features
            y: Labels
            cv (int): Number of folds
            
        Returns:
            dict: Cross-validation scores
        """
        print(f"\n🔄 Performing {cv}-fold cross-validation...")
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        
        print(f"✅ Cross-validation scores: {scores}")
        print(f"   Mean accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std()
        }
    
    def save_model(self, model_dir='models'):
        """
        Save trained model and vectorizer
        
        Args:
            model_dir (str): Directory to save models
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, 'sentiment_model.pkl')
        joblib.dump(self.model, model_path)
        
        # Save vectorizer
        vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
        joblib.dump(self.vectorizer, vectorizer_path)
        
        # Save label encoder
        encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
        joblib.dump(self.label_encoder, encoder_path)
        
        print(f"\n💾 Model saved successfully!")
        print(f"   Model: {model_path}")
        print(f"   Vectorizer: {vectorizer_path}")
        print(f"   Label Encoder: {encoder_path}")
    
    def load_model(self, model_dir='models'):
        """
        Load saved model
        
        Args:
            model_dir (str): Directory containing models
        """
        try:
            self.model = joblib.load(os.path.join(model_dir, 'sentiment_model.pkl'))
            self.vectorizer = joblib.load(os.path.join(model_dir, 'vectorizer.pkl'))
            self.label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
            
            print("✅ Model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def get_feature_importance(self, top_n=20):
        """
        Get top important features (for logistic regression)
        
        Args:
            top_n (int): Number of top features to return
            
        Returns:
            dict: Feature importance by class
        """
        if not isinstance(self.model, LogisticRegression):
            print("⚠️ Feature importance only available for Logistic Regression")
            return None
        
        feature_names = self.vectorizer.get_feature_names_out()
        
        importance_dict = {}
        
        for i, class_name in enumerate(self.label_encoder.classes_):
            if len(self.label_encoder.classes_) == 2 and i == 1:
                # Binary classification
                coef = self.model.coef_[0]
            else:
                coef = self.model.coef_[i]
            
            # Get top positive and negative features
            top_positive_idx = np.argsort(coef)[-top_n:][::-1]
            top_negative_idx = np.argsort(coef)[:top_n]
            
            importance_dict[class_name] = {
                'positive': [(feature_names[idx], coef[idx]) for idx in top_positive_idx],
                'negative': [(feature_names[idx], coef[idx]) for idx in top_negative_idx]
            }
        
        return importance_dict


# Test function
if __name__ == "__main__":
    import os
    
    # Load processed data
    if not os.path.exists('data/processed_tweets.csv'):
        print("❌ No processed data found!")
        print("Run preprocessor first!")
        exit()
    
    print("="*60)
    print("SENTIMENT ANALYSIS MODEL TRAINING")
    print("="*60)
    
    # Load data
    df = pd.read_csv('data/processed_tweets.csv')
    print(f"\n✅ Loaded {len(df)} processed tweets")
    print(f"\n📊 Sentiment distribution:")
    print(df['sentiment'].value_counts())
    
    # Initialize classifier
    classifier = SentimentClassifier(max_features=5000)
    
    # Prepare data
    X_train, X_test, y_train, y_test = classifier.prepare_data(df)
    
    # Train and compare models
    models = ['logistic', 'naive_bayes', 'svm']
    results = []
    
    for model_type in models:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()} model")
        print('='*60)
        
        classifier.train_model(X_train, y_train, model_type)
        accuracy, report = classifier.evaluate_model(X_test, y_test)
        
        results.append({
            'model': model_type,
            'accuracy': accuracy
        })
    
    # Display comparison
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print('='*60)
    
    results_df = pd.DataFrame(results).sort_values('accuracy', ascending=False)
    print(results_df.to_string(index=False))
    
    # Train best model
    best_model = results_df.iloc[0]['model']
    print(f"\n🏆 Best model: {best_model.upper()}")
    
    classifier.train_model(X_train, y_train, best_model)
    classifier.evaluate_model(X_test, y_test)
    
    # Save model
    classifier.save_model()
    
    # Plot confusion matrix
    classifier.plot_confusion_matrix(X_test, y_test)
    
    print("\n✨ Training completed!")