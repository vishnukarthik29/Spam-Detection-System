import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import re
import string
import seaborn as sns
import matplotlib.pyplot as plt

class SpamDetector:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.is_trained = False
        
    def preprocess_text(self, text):
        """
        Simple text preprocessing without NLTK
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove digits
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove common English stopwords manually
        common_stopwords = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
            'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
            'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
            'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
            'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
            'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 'above', 
            'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
            'further', 'then', 'once'
        }
        
        # Remove stopwords
        words = text.split()
        text = ' '.join([word for word in words if word not in common_stopwords])
        
        return text
    
    def load_and_prepare_data(self, url):
        """
        Load and prepare the spam dataset
        """
        try:
            # Load dataset
            df = pd.read_csv(url, encoding="latin-1")
            
            # Keep only the first two columns and rename them
            df = df.iloc[:, :2]
            df.columns = ['label', 'message']
            
            # Remove any missing values
            df = df.dropna()
            
            # Convert labels to binary (ham=0, spam=1)
            df['label'] = df['label'].map({'ham': 0, 'spam': 1})
            
            return df
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
    
    def train_model(self, df):
        """
        Train the Naive Bayes model
        """
        # Preprocess messages
        df['processed_message'] = df['message'].apply(self.preprocess_text)
        
        # Split the data
        X = df['processed_message']
        y = df['label']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Vectorize the text data using CountVectorizer
        self.vectorizer = CountVectorizer(
            max_features=5000, 
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2)  # Use both unigrams and bigrams
        )
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)
        
        # Train Naive Bayes model
        self.model = MultinomialNB()
        self.model.fit(X_train_vectorized, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_vectorized)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        self.is_trained = True
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    def predict(self, message):
        """
        Predict if a message is spam or ham
        """
        if not self.is_trained:
            return None, None
        
        # Preprocess the message
        processed_message = self.preprocess_text(message)
        
        # Vectorize the message
        message_vectorized = self.vectorizer.transform([processed_message])
        
        # Make prediction
        prediction = self.model.predict(message_vectorized)[0]
        prediction_proba = self.model.predict_proba(message_vectorized)[0]
        
        return prediction, prediction_proba

# Streamlit App
def main():
    st.set_page_config(
        page_title="Spam Detection System",
        page_icon="📧",
        layout="wide"
    )
    
    st.title("📧 SMS Spam Detection System")
    st.markdown("*Using Naive Bayes Machine Learning Algorithm*")
    
    # Initialize the spam detector
    if 'spam_detector' not in st.session_state:
        st.session_state.spam_detector = SpamDetector()
    
    spam_detector = st.session_state.spam_detector
    
    # Sidebar for model training
    with st.sidebar:
        st.header("📊 Model Training")
        
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Loading and training model..."):
                # Load data
                url = "https://raw.githubusercontent.com/codebasics/py/master/ML/14_naive_bayes/spam.csv"
                df = spam_detector.load_and_prepare_data(url)
                
                if df is not None:
                    # Train model
                    results = spam_detector.train_model(df)
                    
                    st.success("✅ Model trained successfully!")
                    st.metric("Accuracy", f"{results['accuracy']:.4f}")
                    
                    # Store results in session state
                    st.session_state.training_results = results
                    st.session_state.dataset = df
        
        if spam_detector.is_trained:
            st.success("🎯 Model is ready for predictions!")
        else:
            st.warning("⚠️ Please train the model first")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔍 Spam Detection")
        
        # Text input for message
        message = st.text_area(
            "Enter your message:",
            placeholder="Type your SMS message here...",
            height=150
        )
        
        if st.button("🕵️ Predict", type="primary") and message:
            if not spam_detector.is_trained:
                st.error("❌ Please train the model first using the sidebar!")
            else:
                with st.spinner("Analyzing message..."):
                    prediction, probabilities = spam_detector.predict(message)
                    
                    if prediction is not None:
                        # Display results
                        if prediction == 1:
                            st.error("🚨 **SPAM DETECTED**")
                            spam_confidence = probabilities[1] * 100
                            st.progress(spam_confidence / 100)
                            st.write(f"Spam Confidence: **{spam_confidence:.2f}%**")
                        else:
                            st.success("✅ **HAM (Not Spam)**")
                            ham_confidence = probabilities[0] * 100
                            st.progress(ham_confidence / 100)
                            st.write(f"Ham Confidence: **{ham_confidence:.2f}%**")
                        
                        # Show probability breakdown
                        st.subheader("📈 Prediction Probabilities")
                        prob_df = pd.DataFrame({
                            'Class': ['Ham', 'Spam'],
                            'Probability': [probabilities[0], probabilities[1]]
                        })
                        st.bar_chart(prob_df.set_index('Class'))
    
    with col2:
        st.header("📋 Model Information")
        
        if spam_detector.is_trained and 'training_results' in st.session_state:
            results = st.session_state.training_results
            
            # Display metrics
            st.metric("Model Accuracy", f"{results['accuracy']:.4f}")
            
            # Show confusion matrix
            st.subheader("🔥 Confusion Matrix")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(
                results['confusion_matrix'],
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'],
                ax=ax
            )
            ax.set_title('Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
            
        else:
            st.info("Train the model to see performance metrics")
    
    # Dataset overview
    if 'dataset' in st.session_state:
        st.header("📊 Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        
        df = st.session_state.dataset
        
        with col1:
            st.metric("Total Messages", len(df))
        
        with col2:
            spam_count = (df['label'] == 1).sum()
            st.metric("Spam Messages", spam_count)
        
        with col3:
            ham_count = (df['label'] == 0).sum()
            st.metric("Ham Messages", ham_count)
        
        # Class distribution
        st.subheader("📈 Class Distribution")
        class_dist = df['label'].value_counts()
        class_dist.index = ['Ham', 'Spam']
        st.bar_chart(class_dist)
        
        # Sample messages
        with st.expander("👀 View Sample Messages"):
            st.subheader("Sample Ham Messages")
            ham_samples = df[df['label'] == 0]['message'].head(3)
            for i, msg in enumerate(ham_samples, 1):
                st.write(f"**{i}.** {msg}")
            
            st.subheader("Sample Spam Messages")
            spam_samples = df[df['label'] == 1]['message'].head(3)
            for i, msg in enumerate(spam_samples, 1):
                st.write(f"**{i}.** {msg}")

if __name__ == "__main__":
    main()