"""
Twitter Sentiment Analysis - Streamlit App
Auto-setup enabled for better user experience
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Import utilities and modules
from utils.setup_helper import SetupHelper
from modules.data_generator import DataGenerator
from modules.preprocessor import TextPreprocessor
from modules.model_trainer import SentimentClassifier
from modules.predictor import SentimentPredictor

# Page config
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1DA1F2;
        padding: 20px;
        background: linear-gradient(90deg, #1DA1F2 0%, #14171A 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stButton>button {
        background-color: #1DA1F2;
        color: white;
        font-weight: bold;
        border-radius: 20px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #0d8bd9;
    }
    .success-box {
        padding: 20px;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'setup_complete' not in st.session_state:
    st.session_state.setup_complete = False
if 'collected_tweets' not in st.session_state:
    st.session_state.collected_tweets = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

# AUTO-SETUP ON FIRST RUN
@st.cache_resource
def auto_setup():
    """Run automatic setup once"""
    setup = SetupHelper()
    status = setup.get_status()
    
    if not status['ready']:
        with st.spinner('🚀 Setting up project for first use... This will take ~30 seconds'):
            setup.setup_project()
    
    return setup.get_status()

# Run auto-setup
project_status = auto_setup()

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/000000/twitter--v1.png", width=100)
st.sidebar.title("🐦 Twitter Sentiment Analysis")

# Navigation
selected = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📊 Data", "🧹 Preprocess", "🤖 Model", 
     "📈 Visualizations", "🔮 Predict", "📑 Report"]
)

st.sidebar.markdown("---")

# Project Status in Sidebar
st.sidebar.markdown("### 📊 Project Status")
if project_status['has_data']:
    st.sidebar.success("✅ Data Ready")
else:
    st.sidebar.warning("⏳ No Data")

if project_status['has_processed_data']:
    st.sidebar.success("✅ Data Preprocessed")
else:
    st.sidebar.warning("⏳ Not Preprocessed")

if project_status['has_model']:
    st.sidebar.success("✅ Model Trained")
else:
    st.sidebar.warning("⏳ Model Not Trained")

# Main content based on selection
if selected == "🏠 Home":
    st.markdown('<h1 class="main-header">Twitter Sentiment Analysis</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Status overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if project_status['has_data']:
            df = pd.read_csv('data/raw_tweets.csv')
            st.metric("Total Tweets", len(df))
        else:
            st.metric("Total Tweets", "0")
    
    with col2:
        if project_status['has_processed_data']:
            df = pd.read_csv('data/processed_tweets.csv')
            st.metric("Processed", len(df))
        else:
            st.metric("Processed", "0")
    
    with col3:
        if project_status['has_model']:
            st.metric("Model Status", "✅ Ready")
        else:
            st.metric("Model Status", "⏳ Training...")
    
    with col4:
        if project_status['ready']:
            st.metric("Project Status", "✅ Ready")
        else:
            st.metric("Project Status", "⚙️ Setting up...")
    
    st.markdown("---")
    
    # Welcome message
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Welcome to Sentiment Analysis Dashboard
        
        This application performs **AI-powered sentiment analysis** on text data.
        
        #### ✨ Features:
        
        1. **📊 Sample Data** - Pre-loaded realistic tweets
        2. **🧹 Text Preprocessing** - Clean and prepare text
        3. **🤖 ML Model** - Trained and ready to use
        4. **📈 Visualizations** - Charts, word clouds, analytics
        5. **🔮 Real-time Predictions** - Analyze any text instantly
        
        #### 🚀 Quick Start:
        
        """)
        
        if project_status['ready']:
            st.success("""
            **✅ Everything is ready!**
            
            You can start using the app right away:
            - Go to **🔮 Predict** to analyze text
            - Check **📈 Visualizations** for insights
            - View **📑 Report** for complete analysis
            """)
        else:
            st.info("""
            **⚙️ Auto-setup in progress...**
            
            The app is setting itself up automatically.
            Refresh the page in a moment.
            """)
    
    with col2:
        st.image("https://img.icons8.com/clouds/300/000000/artificial-intelligence.png")
        
        st.markdown("""
        ### 📊 Dataset Info
        """)
        
        if project_status['has_processed_data']:
            df = pd.read_csv('data/processed_tweets.csv')
            sentiment_counts = df['sentiment'].value_counts()
            
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Sentiment Distribution",
                color=sentiment_counts.index,
                color_discrete_map={
                    'Positive': '#2ecc71',
                    'Negative': '#e74c3c',
                    'Neutral': '#95a5a6'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Loading data...")

elif selected == "📊 Data":
    st.header("📊 Dataset Overview")
    
    if project_status['has_data']:
        df = pd.read_csv('data/raw_tweets.csv')
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Tweets", len(df))
        col2.metric("Avg Likes", f"{df['likes'].mean():.1f}")
        col3.metric("Avg Retweets", f"{df['retweets'].mean():.1f}")
        col4.metric("Avg Replies", f"{df['replies'].mean():.1f}")
        
        st.markdown("---")
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📋 Raw Data", "📊 Statistics", "🔄 Refresh Data"])
        
        with tab1:
            st.subheader("Sample Tweets")
            st.dataframe(df.head(20), use_container_width=True)
            
            # Download option
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Dataset (CSV)",
                csv,
                "tweets_data.csv",
                "text/csv"
            )
        
        with tab2:
            st.subheader("Statistical Summary")
            st.write(df.describe())
            
            # Engagement distribution
            fig = px.box(df, y=['likes', 'retweets', 'replies'], 
                        title="Engagement Metrics Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Generate New Sample Data")
            st.info("This will create a fresh set of sample tweets")
            
            num_tweets = st.slider("Number of tweets to generate", 50, 300, 100)
            
            if st.button("🔄 Generate New Data"):
                with st.spinner("Generating new sample data..."):
                    generator = DataGenerator()
                    df_new = generator.create_sample_data(num_tweets)
                    st.success(f"✅ Generated {len(df_new)} new tweets!")
                    st.info("⚠️ Note: You'll need to re-run preprocessing and model training")
                    st.rerun()
    else:
        st.warning("No data available")
        if st.button("📊 Generate Sample Data"):
            with st.spinner("Creating sample data..."):
                generator = DataGenerator()
                generator.create_sample_data()
                st.success("✅ Data created!")
                st.rerun()

elif selected == "🧹 Preprocess":
    st.header("🧹 Data Preprocessing")
    
    if not project_status['has_data']:
        st.error("❌ No data available. Generate sample data first from the 📊 Data section.")
        st.stop()
    
    df = pd.read_csv('data/raw_tweets.csv')
    st.info(f"📊 Ready to process {len(df)} tweets")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Preprocessing Configuration")
        
        remove_stopwords = st.checkbox("🗑️ Remove Stopwords", value=True)
        lemmatize = st.checkbox("📝 Lemmatize Words", value=True)
        
        if st.button("🚀 Start Preprocessing"):
            with st.spinner("Processing..."):
                progress = st.progress(0)
                
                preprocessor = TextPreprocessor()
                
                progress.progress(20)
                df['cleaned_text'] = df['text'].apply(preprocessor.clean_text)
                
                if remove_stopwords:
                    progress.progress(40)
                    df['cleaned_text'] = df['cleaned_text'].apply(preprocessor.remove_stopwords)
                
                if lemmatize:
                    progress.progress(60)
                    df['cleaned_text'] = df['cleaned_text'].apply(preprocessor.lemmatize_text)
                
                progress.progress(80)
                df['sentiment'] = df['text'].apply(preprocessor.get_sentiment_label)
                
                progress.progress(90)
                os.makedirs('data', exist_ok=True)
                df.to_csv('data/processed_tweets.csv', index=False)
                
                progress.progress(100)
                st.success("✅ Preprocessing complete!")
                st.rerun()
    
    with col2:
        st.info("""
        ### 📋 Steps:
        
        1. Clean text
        2. Remove URLs
        3. Remove mentions
        4. Remove stopwords
        5. Lemmatize
        6. Analyze sentiment
        """)
    
    # Show results if processed
    if project_status['has_processed_data']:
        st.markdown("---")
        st.subheader("✅ Processed Data Preview")
        
        df_processed = pd.read_csv('data/processed_tweets.csv')
        
        for i in range(min(3, len(df_processed))):
            with st.expander(f"Sample {i+1}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original:**")
                    st.write(df_processed.iloc[i]['text'])
                with col2:
                    st.markdown("**Cleaned:**")
                    st.write(df_processed.iloc[i]['cleaned_text'])
                    st.markdown(f"**Sentiment:** {df_processed.iloc[i]['sentiment']}")

elif selected == "🤖 Model":
    st.header("🤖 Machine Learning Model")
    
    if not project_status['has_processed_data']:
        st.error("❌ No preprocessed data. Complete preprocessing first.")
        st.stop()
    
    df = pd.read_csv('data/processed_tweets.csv')
    
    if project_status['has_model']:
        st.success("✅ Model is already trained and ready!")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Model Type", "Logistic Regression")
        col2.metric("Status", "✅ Trained")
        col3.metric("Features", "5,000")
        
        if st.checkbox("🔄 Retrain Model"):
            st.warning("This will replace the current model")
            
            if st.button("Confirm Retrain"):
                with st.spinner("Training new model..."):
                    classifier = SentimentClassifier()
                    X_train, X_test, y_train, y_test = classifier.prepare_data(df)
                    classifier.train_model(X_train, y_train, 'logistic')
                    accuracy, report = classifier.evaluate_model(X_test, y_test)
                    classifier.save_model()
                    
                    st.success(f"✅ Model retrained! Accuracy: {accuracy:.2%}")
                    st.text(report)
                    st.rerun()
    else:
        st.info("Training model automatically...")
        
        with st.spinner("Training model... (~30 seconds)"):
            classifier = SentimentClassifier()
            X_train, X_test, y_train, y_test = classifier.prepare_data(df)
            classifier.train_model(X_train, y_train, 'logistic')
            accuracy, report = classifier.evaluate_model(X_test, y_test)
            classifier.save_model()
            
            st.success(f"✅ Model trained! Accuracy: {accuracy:.2%}")
            st.text(report)
            st.rerun()

elif selected == "📈 Visualizations":
    st.header("📈 Data Visualizations")
    
    if not project_status['has_processed_data']:
        st.error("❌ No processed data available")
        st.stop()
    
    df = pd.read_csv('data/processed_tweets.csv')
    
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "☁️ Word Clouds", "📈 Engagement"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            sentiment_counts = df['sentiment'].value_counts()
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Sentiment Distribution",
                color=sentiment_counts.index,
                color_discrete_map={
                    'Positive': '#2ecc71',
                    'Negative': '#e74c3c',
                    'Neutral': '#95a5a6'
                },
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                x=sentiment_counts.index,
                y=sentiment_counts.values,
                title="Sentiment Counts",
                labels={'x': 'Sentiment', 'y': 'Count'},
                color=sentiment_counts.index,
                color_discrete_map={
                    'Positive': '#2ecc71',
                    'Negative': '#e74c3c',
                    'Neutral': '#95a5a6'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Top words
        st.markdown("---")
        st.subheader("Most Frequent Words")
        
        from collections import Counter
        
        col1, col2, col3 = st.columns(3)
        
        for col, sentiment, color in zip([col1, col2, col3],
                                        ['Positive', 'Negative', 'Neutral'],
                                        ['#2ecc71', '#e74c3c', '#95a5a6']):
            with col:
                text = ' '.join(df[df['sentiment'] == sentiment]['cleaned_text'])
                words = text.split()
                word_counts = Counter(words).most_common(10)
                
                if word_counts:
                    words_df = pd.DataFrame(word_counts, columns=['Word', 'Count'])
                    fig = px.bar(
                        words_df,
                        x='Count',
                        y='Word',
                        orientation='h',
                        title=f"{sentiment}",
                        color_discrete_sequence=[color]
                    )
                    fig.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        sentiment_choice = st.selectbox("Select Sentiment", ['Positive', 'Negative', 'Neutral'])
        
        text = ' '.join(df[df['sentiment'] == sentiment_choice]['cleaned_text'])
        
        if text.strip():
            color_map = {
                'Positive': 'Greens',
                'Negative': 'Reds',
                'Neutral': 'Greys'
            }
            
            wordcloud = WordCloud(
                width=1200,
                height=600,
                background_color='white',
                colormap=color_map[sentiment_choice],
                max_words=100
            ).generate(text)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(df, x='sentiment', y='likes',
                        title="Likes by Sentiment",
                        color='sentiment',
                        color_discrete_map={
                            'Positive': '#2ecc71',
                            'Negative': '#e74c3c',
                            'Neutral': '#95a5a6'
                        })
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df, x='sentiment', y='retweets',
                        title="Retweets by Sentiment",
                        color='sentiment',
                        color_discrete_map={
                            'Positive': '#2ecc71',
                            'Negative': '#e74c3c',
                            'Neutral': '#95a5a6'
                        })
            st.plotly_chart(fig, use_container_width=True)

elif selected == "🔮 Predict":
    st.header("🔮 Sentiment Prediction")
    
    if not project_status['has_model']:
        st.error("❌ Model not trained yet")
        st.info("The model is training automatically. Refresh in a moment.")
        st.stop()
    
    try:
        predictor = SentimentPredictor()
        st.success("✅ Model loaded and ready!")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()
    
    st.markdown("---")
    
    # Prediction interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_text = st.text_area(
            "Enter text to analyze:",
            placeholder="Type or paste text here...",
            height=150
        )
        
        if st.button("🔮 Analyze Sentiment", type="primary"):
            if user_text.strip():
                with st.spinner("Analyzing..."):
                    sentiment, confidence, proba_dict = predictor.predict(user_text)
                
                st.markdown("---")
                
                emoji_map = {
                    'Positive': '😊',
                    'Negative': '😞',
                    'Neutral': '😐'
                }
                
                color_map = {
                    'Positive': '#2ecc71',
                    'Negative': '#e74c3c',
                    'Neutral': '#95a5a6'
                }
                
                st.markdown(f"""
                <div style='background-color: {color_map[sentiment]}; padding: 30px; 
                            border-radius: 15px; text-align: center; margin: 20px 0;'>
                    <h1 style='color: white; margin: 0;'>{emoji_map[sentiment]} {sentiment}</h1>
                    <h3 style='color: white; margin: 10px 0 0 0;'>Confidence: {confidence:.1f}%</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Probability chart
                proba_df = pd.DataFrame({
                    'Sentiment': list(proba_dict.keys()),
                    'Probability': list(proba_dict.values())
                }).sort_values('Probability', ascending=False)
                
                fig = px.bar(
                    proba_df,
                    x='Sentiment',
                    y='Probability',
                    color='Sentiment',
                    color_discrete_map=color_map,
                    text='Probability',
                    title="Probability Distribution"
                )
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Please enter some text")
    
    with col2:
        st.info("""
        ### 💡 Try These Examples:
        
        **Positive:**
        - "I love this! Amazing!"
        - "Best day ever! 🎉"
        
        **Negative:**
        - "This is terrible"
        - "Worst experience"
        
        **Neutral:**
        - "It's okay"
        - "Nothing special"
        """)
        
        if st.button("🎲 Random Example"):
            examples = [
                "I absolutely love this! Best product ever! 🎉",
                "This is terrible. Worst experience.",
                "It's okay. Nothing special.",
                "Python is amazing for data science!",
                "I hate waiting. So frustrating!",
                "The weather is nice today."
            ]
            st.session_state.random_example = np.random.choice(examples)
            st.rerun()

elif selected == "📑 Report":
    st.header("📑 Analysis Report")
    
    if not project_status['has_processed_data']:
        st.error("❌ No data available")
        st.stop()
    
    df = pd.read_csv('data/processed_tweets.csv')
    
    # Executive summary
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Tweets", len(df))
    col2.metric("Positive", len(df[df['sentiment'] == 'Positive']))
    col3.metric("Negative", len(df[df['sentiment'] == 'Negative']))
    col4.metric("Neutral", len(df[df['sentiment'] == 'Neutral']))
    
    st.markdown("---")
    
    # Key insights
    sentiment_pct = df['sentiment'].value_counts(normalize=True) * 100
    dominant = sentiment_pct.idxmax()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Key Insights")
        st.markdown(f"""
        - **Dominant Sentiment**: {dominant} ({sentiment_pct[dominant]:.1f}%)
        - **Total Analyzed**: {len(df):,} tweets
        - **Average Likes**: {df['likes'].mean():.1f}
        - **Average Retweets**: {df['retweets'].mean():.1f}
        - **Most Engaging**: {df.groupby('sentiment')['likes'].mean().idxmax()} tweets
        """)
    
    with col2:
        fig = px.pie(
            values=sentiment_pct.values,
            names=sentiment_pct.index,
            color=sentiment_pct.index,
            color_discrete_map={
                'Positive': '#2ecc71',
                'Negative': '#e74c3c',
                'Neutral': '#95a5a6'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Download options
    st.markdown("---")
    st.subheader("📥 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📊 Download Full Data (CSV)",
            csv,
            "sentiment_analysis.csv",
            "text/csv"
        )
    
    with col2:
        summary = f"""SENTIMENT ANALYSIS REPORT
========================

Total Tweets: {len(df)}
Positive: {len(df[df['sentiment'] == 'Positive'])} ({sentiment_pct.get('Positive', 0):.1f}%)
Negative: {len(df[df['sentiment'] == 'Negative'])} ({sentiment_pct.get('Negative', 0):.1f}%)
Neutral: {len(df[df['sentiment'] == 'Neutral'])} ({sentiment_pct.get('Neutral', 0):.1f}%)

Average Engagement:
- Likes: {df['likes'].mean():.1f}
- Retweets: {df['retweets'].mean():.1f}
- Replies: {df['replies'].mean():.1f}
"""
        st.download_button(
            "📄 Download Summary (TXT)",
            summary,
            "summary.txt",
            "text/plain"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🐦 Twitter Sentiment Analysis | Built with Streamlit & Machine Learning</p>
</div>
""", unsafe_allow_html=True)