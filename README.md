# 🐦 Twitter Sentiment Analysis 

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)

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
   git clone https://github.com/ItsAnshumanPattanayak/Sentiment-Analysis-on-Social-Media-Tweets-Project.git
   cd Sentiment-Analysis-on-Social-Media-Tweets-Project
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Start the app**

   ```bash
   streamlit run app.py
   ```

## 📊 Usage

### Option 1: With Twitter API

1. Get Twitter API credentials from the Twitter Developer Portal.
2. Copy `.env.example` to `.env` and add the credentials.
3. Run the app and use the "📊 Collect Tweets" section.

### Option 2: With Sample Data (No API needed)

1. Generate sample data:

   ```bash
   python create_sample_data.py
   ```

2. Start the app:

   ```bash
   streamlit run app.py
   ```

3. Skip to the "🧹 Preprocess" section.

### Option 3: With Kaggle Dataset

1. Download the Sentiment140 dataset.
2. Run `python load_kaggle_data.py` if you add that loader script.
3. Start the app.

## 🎓 Workflow

1. Collect tweets from the Twitter API or sample data.
2. Preprocess text by cleaning noise and preparing sentiment labels.
3. Train ML models, with Logistic Regression as the recommended default.
4. View charts, word clouds, and analytics.
5. Test new tweets or batch process CSV files.
6. Generate a comprehensive analysis report.

## 📈 Model Performance

- **Algorithm**: Logistic Regression with TF-IDF
- **Accuracy**: ~85-90% (varies by dataset)
- **Features**: 5,000 top words with bigrams
- **Classes**: Positive, Negative, Neutral

## 🖼️ Screenshots

### Home Dashboard

<img width="1418" height="762" alt="image" src="https://github.com/user-attachments/assets/8b84a708-0778-4f4e-91af-b16ad6b6d1b2" />

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/amazing-feature`).
3. Commit your changes (`git commit -m 'Add amazing feature'`).
4. Push to the branch (`git push origin feature/amazing-feature`).
5. Open a pull request.

## 🙏 Acknowledgments

- Tweepy - Twitter API library
- Streamlit - Web app framework
- Scikit-learn - Machine learning library
- NLTK - Natural language processing

## 📧 Contact

anshumanpattanayak931@gmail.com

Project Link: https://github.com/ItsAnshumanPattanayak/Sentiment-Analysis-on-Social-Media-Tweets-Project

## 🐛 Known Issues

- Twitter API Free tier doesn't support tweet search.
- Solution: Use sample data or a Kaggle dataset.

## 🔮 Future Enhancements

- Add more ML models (Random Forest, Neural Networks)
- Real-time streaming analysis
- Multi-language support
- Export reports to PDF
- Docker containerization
- API endpoint for predictions
- Advanced visualizations, such as network graphs
 
## ⭐ Star This Repo

If you found this project helpful, please give it a star! ⭐

