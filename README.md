📧 SMS Spam Detection System
A machine learning web application built with Streamlit and Naive Bayes algorithm to detect spam SMS messages in real-time.
🚀 Features

Real-time Spam Detection: Input any SMS message and get instant predictions
Interactive Web Interface: User-friendly Streamlit dashboard
Model Performance Metrics: View accuracy, confusion matrix, and classification reports
Data Visualization: Dataset overview and prediction probability charts
No External Dependencies: Simple setup without NLTK complications

🛠️ Technologies Used

Python 3.7+
Streamlit - Web interface
Scikit-learn - Machine learning algorithms
Pandas - Data manipulation
Matplotlib & Seaborn - Data visualization
CountVectorizer - Text vectorization

📊 Model Details

Algorithm: Multinomial Naive Bayes
Vectorization: CountVectorizer with n-grams (1-2)
Dataset: SMS Spam Collection Dataset (5,572 messages)
Accuracy: ~95-97%
Features: 5,000 most important words and phrases

🏗️ Project Structure
sms-spam-detection/
│
├── app.py # Main Streamlit application
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── .gitignore # Git ignore rules
├── screenshots/ # App screenshots
│ ├── main_interface.png
│ └── prediction_result.png
└── docs/ # Additional documentation
└── USAGE.md # Detailed usage guide
🚀 Quick Start

1. Clone the Repository
   bashgit clone https://github.com/yourusername/sms-spam-detection.git
   cd sms-spam-detection
2. Install Dependencies
   bashpip install -r requirements.txt
3. Run the Application
   bashstreamlit run app.py
4. Open in Browser
   The app will automatically open at http://localhost:8501
   💡 How to Use

Train the Model: Click "🚀 Train Model" in the sidebar
Enter Message: Type any SMS message in the text area
Get Prediction: Click "🕵️ Predict" to see if it's spam or ham
View Results: Check confidence scores and probability breakdown

📈 Model Performance
The Naive Bayes classifier achieves excellent performance on the SMS Spam Collection dataset:

Accuracy: 95-97%
Precision: High for both spam and ham detection
Recall: Effective spam detection with minimal false positives
F1-Score: Balanced performance across both classes

📊 Dataset Information

Source: SMS Spam Collection Dataset
Total Messages: 5,572
Spam Messages: ~747 (13.4%)
Ham Messages: ~4,825 (86.6%)
Languages: English

🔧 Customization
You can easily modify the model parameters:
python# In the train_model method
self.vectorizer = CountVectorizer(
max_features=5000, # Number of features
stop_words='english', # Remove common words
lowercase=True, # Convert to lowercase
ngram_range=(1, 2) # Use unigrams and bigrams
)
📱 Screenshots
Main Interface
Show Image
Prediction Results
Show Image
🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.
Development Setup

Fork the repository
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
🎯 Future Enhancements

Add email spam detection
Implement deep learning models (LSTM, BERT)
Add multi-language support
Create REST API endpoints
Add user feedback mechanism
Implement model retraining functionality

👥 Authors

Your Name - Initial work - YourUsername

🙏 Acknowledgments

SMS Spam Collection Dataset from UCI Machine Learning Repository
Streamlit team for the amazing framework
Scikit-learn contributors for machine learning tools

📞 Support
If you have any questions or need help, please:

Check the Usage Guide
Open an Issue
Contact me at your.email@example.com

⭐ If you found this project helpful, please give it a star!
