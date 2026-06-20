# 🐦 Twitter Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A complete Machine Learning pipeline for sentiment analysis on Twitter data using NLP and interactive web interface.

![Project Demo](https://via.placeholder.com/800x400.png?text=Add+Screenshot+Here)

## 🎯 Features

- 📊 **Tweet Collection** - Fetch tweets using Twitter API v2
- 🧹 **Data Preprocessing** - Clean text, remove stopwords, lemmatization
- 🤖 **ML Models** - Train Logistic Regression, Naive Bayes, SVM
- 📈 **Visualizations** - Interactive charts, word clouds, analytics
- 🔮 **Real-time Prediction** - Analyze sentiment of new tweets
- 📑 **Reports** - Generate comprehensive analysis reports

## 🛠️ Tech Stack

- **Backend**: Python 3.10+
- **Web Framework**: Streamlit
- **ML/NLP**: Scikit-learn, NLTK, TextBlob
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, WordCloud, Seaborn
- **API**: Tweepy (Twitter API v2)

## 📁 Project Structure
sentiment-analysis-twitter/
├── app.py # Main Streamlit application
├── modules/ # Core functionality
│ ├── init.py
│ ├── tweet_collector.py # Tweet collection
│ ├── preprocessor.py # Text preprocessing
│ ├── model_trainer.py # ML model training
│ └── predictor.py # Sentiment prediction
├── data/ # Data files (auto-generated)
├── models/ # Trained models (auto-generated)
├── visualizations/ # Generated charts (auto-generated)
├── requirements.txt # Python dependencies
├── .env.example # Environment variables template
├── .gitignore # Git ignore rules
└── README.md # This file

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Twitter Developer Account (optional - can use sample data)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/twitter-sentiment-analysis.git
   cd twitter-sentiment-analysis

📊 Usage

Option 1: With Twitter API

Get Twitter API credentials from Twitter Developer Portal
Add credentials to .env file
Run app and use "📊 Collect Tweets" section

Option 2: With Sample Data (No API needed)

Run: python create_sample_data.py
Start app: streamlit run app.py
Skip to "🧹 Preprocess" section

Option 3: With Kaggle Dataset

Download Sentiment140 dataset
Run: python load_kaggle_data.py
Start the app


🎓 Workflow

1. Collect Tweets → 2. Preprocess → 3. Train Model → 4. Visualize → 5. Predict
Collect Tweets - Gather data from Twitter API or use sample data
Preprocess - Clean text, remove noise, analyze sentiment
Train Model - Train ML models (Logistic Regression recommended)
Visualizations - View charts, word clouds, analytics
Predict - Test on new tweets or batch process CSV files
Report - Generate comprehensive analysis reports

📈 Model Performance

Algorithm: Logistic Regression with TF-IDF
Accuracy: ~85-90% (varies by dataset)
Features: 5,000 top words with bigrams
Classes: Positive, Negative, Neutral

🖼️ Screenshots

Home Dashboard
<img width="1418" height="762" alt="image" src="https://github.com/user-attachments/assets/8b84a708-0778-4f4e-91af-b16ad6b6d1b2" />

🤝 Contributing
Contributions are welcome! Please follow these steps:

Fork the repository
Create a new branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request
📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Tweepy - Twitter API library
Streamlit - Web app framework
Scikit-learn - Machine learning library
NLTK - Natural language processing
📧 Contact
anshumanpattanayak931@gmail.com

Project Link: https://github.com/YOUR_USERNAME/twitter-sentiment-analysis

🐛 Known Issues
Twitter API Free tier doesn't support tweet search (requires $100/month Basic plan)
Solution: Use sample data or Kaggle dataset (both free)

🔮 Future Enhancements
 Add more ML models (Random Forest, Neural Networks)
 Real-time streaming analysis
 Multi-language support
 Export reports to PDF
 Docker containerization
 API endpoint for predictions
 Advanced visualizations (network graphs
 
⭐ Star This Repo
If you found this project helpful, please give it a star! ⭐


