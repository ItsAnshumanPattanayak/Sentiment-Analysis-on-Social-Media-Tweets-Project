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

# Import custom modules
from modules.tweet_collector import TweetCollector
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
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'collected_tweets' not in st.session_state:
    st.session_state.collected_tweets = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Sidebar navigation (using built-in selectbox)
st.sidebar.image("https://img.icons8.com/color/96/000000/twitter--v1.png", width=100)
st.sidebar.title("🐦 Twitter Sentiment Analysis")

# Simple navigation menu
selected = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📊 Collect Tweets", "🧹 Preprocess", "🤖 Train Model", 
     "📈 Visualizations", "🔮 Predict", "📑 Report"]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Pro Tip:** Follow the steps in order for best results!")

# Display current status
st.sidebar.markdown("### 📊 Project Status")
if os.path.exists('data/raw_tweets.csv'):
    st.sidebar.success("✅ Tweets Collected")
else:
    st.sidebar.warning("⏳ No Tweets Yet")

if os.path.exists('data/processed_tweets.csv'):
    st.sidebar.success("✅ Data Preprocessed")
else:
    st.sidebar.warning("⏳ Not Preprocessed")

if os.path.exists('models/sentiment_model.pkl'):
    st.sidebar.success("✅ Model Trained")
else:
    st.sidebar.warning("⏳ Model Not Trained")

# Main content
if selected == "🏠 Home":
    st.markdown('<h1 class="main-header">Twitter Sentiment Analysis Dashboard</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("### 📊 Step 1: Collect\nGather tweets from Twitter API")
    with col2:
        st.warning("### 🧹 Step 2: Preprocess\nClean and prepare data")
    with col3:
        st.success("### 🤖 Step 3: Train\nBuild ML model")
    
    st.markdown("---")
    
    # Project Overview
    st.header("📋 Project Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 What This App Does:
        
        This is a **complete Machine Learning pipeline** for sentiment analysis on Twitter data:
        
        1. **📊 Data Collection**: Fetch real tweets using Twitter API
        2. **🧹 Preprocessing**: Clean text, remove stopwords, lemmatization
        3. **🤖 Model Training**: Train ML models (Logistic Regression, Naive Bayes, SVM)
        4. **📈 Visualization**: Interactive charts, word clouds, analytics
        5. **🔮 Prediction**: Real-time sentiment prediction on new tweets
        
        ### 🛠️ Technologies Used:
        - **API**: Twitter API v2 (Tweepy)
        - **NLP**: NLTK, TextBlob
        - **ML**: Scikit-learn (TF-IDF, Logistic Regression)
        - **Visualization**: Plotly, Matplotlib, WordCloud
        - **Web App**: Streamlit
        """)
    
    with col2:
        st.image("https://img.icons8.com/clouds/300/000000/data-analytics.png")
        
        # Quick stats
        if os.path.exists('data/processed_tweets.csv'):
            df = pd.read_csv('data/processed_tweets.csv')
            st.metric("Total Tweets", len(df))
            st.metric("Unique Words", len(set(' '.join(df['cleaned_text']).split())))
            
        if os.path.exists('models/sentiment_model.pkl'):
            st.success("✅ Model Trained!")
        else:
            st.warning("⚠️ Model Not Trained")
    
    st.markdown("---")
    st.info("👈 **Get Started**: Use the sidebar to navigate through different sections!")

elif selected == "📊 Collect Tweets":
    st.header("📊 Collect Tweets from Twitter")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Search Parameters")
        
        # Search query
        query = st.text_input("🔍 Search Query", 
                              placeholder="e.g., Python, AI, Climate Change",
                              help="Enter keywords to search for tweets")
        
        # Number of tweets
        max_tweets = st.slider("📈 Number of Tweets", 100, 2000, 500, 100)
        
        # Collect button
        if st.button("🚀 Collect Tweets", key="collect_btn"):
            if not query:
                st.error("❌ Please enter a search query!")
            else:
                try:
                    collector = TweetCollector()
                    
                    with st.spinner(f"🔍 Searching for tweets about '{query}'..."):
                        df = collector.collect_tweets(query, max_tweets)
                    
                    if df is not None and len(df) > 0:
                        st.session_state.collected_tweets = df
                        st.success(f"✅ Successfully collected {len(df)} tweets!")
                    else:
                        st.error("❌ No tweets found. Try a different query.")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("💡 Make sure your Twitter API credentials are set in .env file")
    
    with col2:
        st.info("""
        ### 💡 Tips:
        - Use popular hashtags
        - Try trending topics
        - Be specific with keywords
        - More tweets = better model
        
        ### 📝 Examples:
        - `#Python`
        - `Climate Change`
        - `AI Technology`
        - `Movie Reviews`
        """)
    
    # Display collected tweets
    if st.session_state.collected_tweets is not None:
        st.markdown("---")
        st.subheader("📊 Collected Tweets Preview")
        
        df = st.session_state.collected_tweets
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Tweets", len(df))
        col2.metric("Avg Likes", f"{df['likes'].mean():.1f}")
        col3.metric("Avg Retweets", f"{df['retweets'].mean():.1f}")
        col4.metric("Avg Replies", f"{df['replies'].mean():.1f}")
        
        # Display dataframe
        st.dataframe(df.head(20), use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download CSV",
            csv,
            "tweets.csv",
            "text/csv",
            key='download-csv'
        )

elif selected == "🧹 Preprocess":
    st.header("🧹 Data Preprocessing")
    
    if st.session_state.collected_tweets is None:
        if os.path.exists('data/raw_tweets.csv'):
            st.info("📂 Loading previously collected tweets...")
            st.session_state.collected_tweets = pd.read_csv('data/raw_tweets.csv')
        else:
            st.warning("⚠️ No tweets collected yet!")
            st.info("👈 Go to 'Collect Tweets' section first")
            st.stop()
    
    df = st.session_state.collected_tweets
    
    st.info(f"📊 Processing {len(df)} tweets")
    
    # Preprocessing options
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Preprocessing Steps")
        
        remove_urls = st.checkbox("🔗 Remove URLs", value=True)
        remove_mentions = st.checkbox("👤 Remove Mentions (@user)", value=True)
        remove_hashtags = st.checkbox("#️⃣ Remove Hashtags", value=False)
        remove_stopwords = st.checkbox("🗑️ Remove Stopwords", value=True)
        lemmatize = st.checkbox("📝 Lemmatize Words", value=True)
        
        if st.button("🚀 Start Preprocessing", key="preprocess_btn"):
            preprocessor = TextPreprocessor()
            
            with st.spinner("🧹 Preprocessing tweets..."):
                progress_bar = st.progress(0)
                
                # Clean text
                progress_bar.progress(20)
                df['cleaned_text'] = df['text'].apply(preprocessor.clean_text)
                
                # Remove stopwords
                if remove_stopwords:
                    progress_bar.progress(40)
                    df['cleaned_text'] = df['cleaned_text'].apply(preprocessor.remove_stopwords)
                
                # Lemmatize
                if lemmatize:
                    progress_bar.progress(60)
                    df['cleaned_text'] = df['cleaned_text'].apply(preprocessor.lemmatize_text)
                
                # Get sentiment
                progress_bar.progress(80)
                df['sentiment'] = df['text'].apply(preprocessor.get_sentiment_label)
                
                progress_bar.progress(100)
                
                # Save
                os.makedirs('data', exist_ok=True)
                df.to_csv('data/processed_tweets.csv', index=False)
                
                st.session_state.processed_data = df
                
                st.success("✅ Preprocessing completed!")
    
    with col2:
        st.info("""
        ### 📋 What happens:
        
        1. Convert to lowercase
        2. Remove URLs
        3. Remove mentions
        4. Remove special chars
        5. Remove stopwords
        6. Lemmatize words
        7. Analyze sentiment
        """)
    
    # Show before/after
    if st.session_state.processed_data is not None:
        st.markdown("---")
        st.subheader("🔍 Before vs After")
        
        df_processed = st.session_state.processed_data
        
        for i in range(min(5, len(df_processed))):
            with st.expander(f"Tweet {i+1}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📝 Original:**")
                    st.write(df_processed.iloc[i]['text'])
                with col2:
                    st.markdown("**✨ Cleaned:**")
                    st.write(df_processed.iloc[i]['cleaned_text'])
                    st.markdown(f"**Sentiment:** {df_processed.iloc[i]['sentiment']}")
        
        # Sentiment distribution
        st.markdown("---")
        st.subheader("📊 Sentiment Distribution")
        
        sentiment_counts = df_processed['sentiment'].value_counts()
        
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Sentiment Distribution",
            color=sentiment_counts.index,
            color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#95a5a6'}
        )
        st.plotly_chart(fig, use_container_width=True)

elif selected == "🤖 Train Model":
    st.header("🤖 Train Sentiment Analysis Model")
    
    if not os.path.exists('data/processed_tweets.csv'):
        st.warning("⚠️ No preprocessed data found!")
        st.info("👈 Complete preprocessing first")
        st.stop()
    
    df = pd.read_csv('data/processed_tweets.csv')
    
    st.info(f"📊 Training on {len(df)} preprocessed tweets")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Model Configuration")
        
        # Model selection
        model_type = st.selectbox(
            "🤖 Select Model",
            ["Logistic Regression", "Naive Bayes", "SVM"],
            help="Choose the machine learning algorithm"
        )
        
        # Test size
        test_size = st.slider("📊 Test Size (%)", 10, 40, 20)
        
        # TF-IDF parameters
        max_features = st.slider("📈 Max Features (TF-IDF)", 1000, 10000, 5000, 1000)
        
        # Train button
        if st.button("🚀 Train Model", key="train_btn"):
            classifier = SentimentClassifier()
            
            with st.spinner("🤖 Training model..."):
                progress_bar = st.progress(0)
                
                # Prepare data
                progress_bar.progress(20)
                X_train, X_test, y_train, y_test = classifier.prepare_data(df, test_size=test_size/100)
                
                # Train model
                progress_bar.progress(40)
                model_map = {
                    'Logistic Regression': 'logistic',
                    'Naive Bayes': 'naive_bayes',
                    'SVM': 'svm'
                }
                classifier.train_model(X_train, y_train, model_map[model_type])
                
                # Evaluate
                progress_bar.progress(70)
                accuracy, report = classifier.evaluate_model(X_test, y_test)
                
                # Save
                progress_bar.progress(90)
                classifier.save_model()
                
                progress_bar.progress(100)
                
                st.session_state.model_trained = True
                
                st.success(f"✅ Model trained successfully! Accuracy: {accuracy:.2%}")
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Model Performance")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{accuracy:.2%}")
                col2.metric("Model Type", model_type)
                col3.metric("Training Samples", X_train.shape[0])
                
                # Classification report
                st.markdown("### 📋 Classification Report")
                st.text(report)
    
    with col2:
        st.info("""
        ### 📚 Models Explained:
        
        **Logistic Regression**
        - Fast training
        - Good baseline
        - Interpretable
        
        **Naive Bayes**
        - Works well with text
        - Probabilistic
        - Fast predictions
        
        **SVM**
        - High accuracy
        - Good for complex data
        - Slower training
        
        ### 💡 Tips:
        - More data = better accuracy
        - Try different models
        - 20% test size is standard
        """)
    
    # Model comparison
    if st.checkbox("📊 Compare All Models"):
        st.markdown("---")
        st.subheader("🏆 Model Comparison")
        
        classifier = SentimentClassifier()
        X_train, X_test, y_train, y_test = classifier.prepare_data(df)
        
        results = []
        
        for model_name, model_key in [('Logistic Regression', 'logistic'), 
                                       ('Naive Bayes', 'naive_bayes'), 
                                       ('SVM', 'svm')]:
            with st.spinner(f"Training {model_name}..."):
                classifier.train_model(X_train, y_train, model_key)
                accuracy, _ = classifier.evaluate_model(X_test, y_test)
                results.append({'Model': model_name, 'Accuracy': accuracy})
        
        results_df = pd.DataFrame(results)
        
        fig = px.bar(
            results_df,
            x='Model',
            y='Accuracy',
            title="Model Accuracy Comparison",
            color='Accuracy',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)

elif selected == "📈 Visualizations":
    st.header("📈 Sentiment Analysis Visualizations")
    
    if not os.path.exists('data/processed_tweets.csv'):
        st.warning("⚠️ No data found!")
        st.info("👈 Collect and preprocess data first")
        st.stop()
    
    df = pd.read_csv('data/processed_tweets.csv')
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "☁️ Word Clouds", "📈 Engagement", "📉 Trends"])
    
    with tab1:
        st.subheader("📊 Sentiment Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment distribution - Pie
            sentiment_counts = df['sentiment'].value_counts()
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Sentiment Distribution",
                color=sentiment_counts.index,
                color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#95a5a6'},
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sentiment distribution - Bar
            fig = px.bar(
                x=sentiment_counts.index,
                y=sentiment_counts.values,
                title="Sentiment Counts",
                labels={'x': 'Sentiment', 'y': 'Count'},
                color=sentiment_counts.index,
                color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#95a5a6'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Top words
        st.markdown("---")
        st.subheader("🔤 Most Frequent Words by Sentiment")
        
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
                        title=f"{sentiment} Words",
                        color_discrete_sequence=[color]
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("☁️ Word Clouds")
        
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
            ax.set_title(f'{sentiment_choice} Sentiment Word Cloud', fontsize=20, fontweight='bold', pad=20)
            
            st.pyplot(fig)
        else:
            st.warning(f"No {sentiment_choice.lower()} tweets found!")
    
    with tab3:
        st.subheader("📈 Engagement Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Likes by sentiment
            fig = px.box(
                df,
                x='sentiment',
                y='likes',
                title="Likes Distribution by Sentiment",
                color='sentiment',
                color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#95a5a6'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Retweets by sentiment
            fig = px.box(
                df,
                x='sentiment',
                y='retweets',
                title="Retweets Distribution by Sentiment",
                color='sentiment',
                color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#95a5a6'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Average engagement
        st.markdown("---")
        avg_engagement = df.groupby('sentiment')[['likes', 'retweets', 'replies']].mean().round(2)
        
        fig = go.Figure()
        for metric in ['likes', 'retweets', 'replies']:
            fig.add_trace(go.Bar(
                name=metric.capitalize(),
                x=avg_engagement.index,
                y=avg_engagement[metric]
            ))
        
        fig.update_layout(
            title="Average Engagement by Sentiment",
            xaxis_title="Sentiment",
            yaxis_title="Average Count",
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("📉 Timeline Trends")
        
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])
            df['date'] = df['created_at'].dt.date
            
            daily_sentiment = df.groupby(['date', 'sentiment']).size().reset_index(name='count')
            
            fig = px.line(
                daily_sentiment,
                x='date',
                y='count',
                color='sentiment',
                title="Sentiment Trends Over Time",
                color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#95a5a6'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📅 Timeline data not available")

elif selected == "🔮 Predict":
    st.header("🔮 Predict Sentiment")
    
    if not os.path.exists('models/sentiment_model.pkl'):
        st.warning("⚠️ Model not trained yet!")
        st.info("👈 Train the model first")
        st.stop()
    
    # Load predictor
    try:
        predictor = SentimentPredictor()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()
    
    st.markdown("---")
    
    # Input methods
    input_method = st.radio("Choose input method:", ["✍️ Type Text", "📁 Upload CSV"])
    
    if input_method == "✍️ Type Text":
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Text input
            user_text = st.text_area(
                "Enter tweet to analyze:",
                placeholder="Type or paste a tweet here...",
                height=150
            )
            
            if st.button("🔮 Predict Sentiment", key="predict_btn"):
                if user_text.strip():
                    with st.spinner("Analyzing..."):
                        sentiment, confidence, proba_dict = predictor.predict(user_text)
                    
                    # Display result
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
                    <div style='background-color: {color_map[sentiment]}; padding: 20px; border-radius: 10px; text-align: center;'>
                        <h1 style='color: white;'>{emoji_map[sentiment]} {sentiment}</h1>
                        <h3 style='color: white;'>Confidence: {confidence:.1f}%</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Probability distribution
                    st.markdown("---")
                    st.subheader("📊 Probability Distribution")
                    
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
                        text='Probability'
                    )
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ Please enter some text!")
        
        with col2:
            st.info("""
            ### 💡 Examples:
            
            **Positive:**
            - "I love this product! Amazing!"
            - "Best day ever! 🎉"
            
            **Negative:**
            - "This is terrible."
            - "Worst experience ever."
            
            **Neutral:**
            - "It's okay."
            - "Nothing special."
            """)
            
            if st.button("Try Random Example"):
                examples = [
                    "I absolutely love this! Best product ever! 🎉",
                    "This is the worst thing I've ever experienced.",
                    "It's okay, nothing special about it.",
                    "Python is an amazing programming language!",
                    "I hate waiting in long queues.",
                    "The weather is nice today."
                ]
                st.session_state.example_text = np.random.choice(examples)
                st.rerun()
    
    else:  # CSV Upload
        st.subheader("📁 Upload CSV File")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            df_upload = pd.read_csv(uploaded_file)
            
            st.subheader("📊 Data Preview")
            st.dataframe(df_upload.head())
            
            # Select text column
            text_column = st.selectbox("Select text column:", df_upload.columns)
            
            if st.button("🔮 Predict All", key="predict_all_btn"):
                with st.spinner("Analyzing all tweets..."):
                    predictions = []
                    confidences = []
                    
                    progress_bar = st.progress(0)
                    
                    for i, text in enumerate(df_upload[text_column]):
                        sentiment, confidence, _ = predictor.predict(str(text))
                        predictions.append(sentiment)
                        confidences.append(confidence)
                        progress_bar.progress((i + 1) / len(df_upload))
                    
                    df_upload['predicted_sentiment'] = predictions
                    df_upload['confidence'] = confidences
                    
                    st.success("✅ Predictions completed!")
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("📊 Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    sentiment_counts = df_upload['predicted_sentiment'].value_counts()
                    col1.metric("Positive", sentiment_counts.get('Positive', 0))
                    col2.metric("Negative", sentiment_counts.get('Negative', 0))
                    col3.metric("Neutral", sentiment_counts.get('Neutral', 0))
                    
                    # Show dataframe
                    st.dataframe(df_upload, use_container_width=True)
                    
                    # Download button
                    csv = df_upload.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Results",
                        csv,
                        "predictions.csv",
                        "text/csv",
                        key='download-predictions'
                    )

elif selected == "📑 Report":
    st.header("📑 Complete Analysis Report")
    
    # Check if all data exists
    has_data = os.path.exists('data/processed_tweets.csv')
    has_model = os.path.exists('models/sentiment_model.pkl')
    
    if not has_data:
        st.warning("⚠️ No data available. Complete the pipeline first!")
        st.stop()
    
    df = pd.read_csv('data/processed_tweets.csv')
    
    # Executive Summary
    st.subheader("📊 Executive Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total Tweets", len(df))
    col2.metric("Positive", len(df[df['sentiment'] == 'Positive']))
    col3.metric("Negative", len(df[df['sentiment'] == 'Negative']))
    col4.metric("Neutral", len(df[df['sentiment'] == 'Neutral']))
    
    # Key Insights
    st.markdown("---")
    st.subheader("🔍 Key Insights")
    
    sentiment_pct = df['sentiment'].value_counts(normalize=True) * 100
    dominant_sentiment = sentiment_pct.idxmax()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        insights = f"""
        ### 📈 Main Findings:
        
        1. **Dominant Sentiment**: {dominant_sentiment} ({sentiment_pct[dominant_sentiment]:.1f}%)
        2. **Total Tweets Analyzed**: {len(df):,}
        3. **Average Engagement**:
           - Likes: {df['likes'].mean():.1f}
           - Retweets: {df['retweets'].mean():.1f}
           - Replies: {df['replies'].mean():.1f}
        
        4. **Most Engaging Sentiment**: {df.groupby('sentiment')['likes'].mean().idxmax()}
        
        ### 💡 Recommendations:
        
        - Focus on {dominant_sentiment.lower()} content
        - Monitor negative feedback closely
        - Engage with top-performing tweets
        """
        st.markdown(insights)
    
    with col2:
        # Sentiment pie chart
        fig = px.pie(
            values=sentiment_pct.values,
            names=sentiment_pct.index,
            title="Sentiment Breakdown",
            color=sentiment_pct.index,
            color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#95a5a6'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Model Performance
    if has_model:
        st.markdown("---")
        st.subheader("🤖 Model Performance")
        
        st.info("""
        - **Algorithm**: Logistic Regression with TF-IDF
        - **Features**: 5,000 top words
        - **Accuracy**: ~85-90% (varies by dataset)
        - **Training Time**: < 1 minute
        """)
    
    # Download Report
    st.markdown("---")
    st.subheader("📥 Export Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📊 Download Full Data (CSV)",
            csv,
            "sentiment_analysis_report.csv",
            "text/csv"
        )
    
    with col2:
        # Summary download
        summary = f"""
        SENTIMENT ANALYSIS REPORT
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
    <p>🐦 Twitter Sentiment Analysis Dashboard | Built with Streamlit</p>
    <p>💡 Powered by Machine Learning & NLP</p>
</div>
""", unsafe_allow_html=True)