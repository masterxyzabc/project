import re
import string
import pickle
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

class SentimentAnalyzer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.tfidf_vectorizer = None
        self.ml_model = None
        self.category_keywords = self._initialize_category_keywords()
        
        # Download NLTK data
        try:
            import nltk
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            import nltk
            nltk.download('punkt_tab')
        
        try:
            import nltk
            nltk.data.find('corpora/stopwords')
        except LookupError:
            import nltk
            nltk.download('stopwords')
            
        try:
            import nltk
            nltk.data.find('corpora/wordnet')
        except LookupError:
            import nltk
            nltk.download('wordnet')
    
    def _initialize_category_keywords(self):
        return {
            'Product Quality': [
                'quality', 'product', 'item', 'broken', 'defective', 'damaged', 
                'poor quality', 'high quality', 'excellent', 'durable', 'sturdy',
                'cheap', 'expensive', 'worth', 'value', 'materials', 'construction'
            ],
            'Delivery': [
                'delivery', 'shipping', 'arrived', 'late', 'fast', 'slow', 'package',
                'courier', 'tracking', 'dispatch', 'arrive', 'shipment', 'logistics'
            ],
            'Customer Service': [
                'service', 'support', 'staff', 'representative', 'help', 'rude',
                'friendly', 'responsive', 'helpful', 'unhelpful', 'agent', 'team'
            ],
            'App Experience': [
                'app', 'application', 'website', 'interface', 'user interface', 'ui',
                'crash', 'slow', 'fast', 'bug', 'glitch', 'navigation', 'design'
            ],
            'Pricing': [
                'price', 'cost', 'expensive', 'cheap', 'affordable', 'overpriced',
                'discount', 'sale', 'offer', 'deal', 'value for money', 'budget'
            ],
            'Other': []
        }
    
    def preprocess_text(self, text):
        """Comprehensive text preprocessing"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Handle emojis (convert to text)
        text = self._handle_emojis(text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Handle negations (not good -> not_good)
        text = self._handle_negations(text)
        
        # Remove punctuation and numbers
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        # Lemmatization
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Remove short words
        tokens = [token for token in tokens if len(token) > 2]
        
        return ' '.join(tokens)
    
    def _handle_emojis(self, text):
        """Convert emojis to text descriptions"""
        emoji_dict = {
            '😊': 'smile',
            '😍': 'love',
            '👍': 'good',
            '👎': 'bad',
            '😢': 'sad',
            '😡': 'angry',
            '🎉': 'celebration',
            '💯': 'perfect',
            '❤️': 'love',
            '⭐': 'star',
            '🌟': 'excellent'
        }
        
        for emoji, description in emoji_dict.items():
            text = text.replace(emoji, f' {description} ')
        
        return text
    
    def _handle_negations(self, text):
        """Handle negations by connecting them with the following word"""
        negations = ['not', 'no', 'never', 'none', 'nothing', 'nowhere', 
                    'neither', 'nobody', 'cannot', "can't", "won't", "don't", 
                    "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't"]
        
        words = text.split()
        result = []
        
        for i, word in enumerate(words):
            if word in negations and i + 1 < len(words):
                result.append(f"{word}_{words[i+1]}")
                # Skip the next word as it's already combined
                continue
            elif i == 0 or words[i-1] not in negations:
                result.append(word)
        
        return ' '.join(result)
    
    def analyze_sentiment_vader(self, text):
        """VADER rule-based sentiment analysis"""
        scores = self.vader_analyzer.polarity_scores(text)
        
        if scores['compound'] >= 0.05:
            sentiment = 'positive'
        elif scores['compound'] <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'compound': scores['compound'],
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu']
        }
    
    def analyze_sentiment_textblob(self, text):
        """TextBlob sentiment analysis"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'polarity': polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    def train_ml_model(self, training_data):
        """Train ML-based sentiment classifier"""
        # Preprocess all texts
        processed_texts = [self.preprocess_text(text) for text in training_data['text']]
        
        # TF-IDF Vectorization
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X = self.tfidf_vectorizer.fit_transform(processed_texts)
        y = training_data['sentiment']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Logistic Regression model
        self.ml_model = LogisticRegression(random_state=42, max_iter=1000)
        self.ml_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.ml_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    def analyze_sentiment_ml(self, text):
        """ML-based sentiment analysis"""
        if not self.ml_model or not self.tfidf_vectorizer:
            raise ValueError("ML model not trained. Call train_ml_model() first.")
        
        processed_text = self.preprocess_text(text)
        text_vector = self.tfidf_vectorizer.transform([processed_text])
        prediction = self.ml_model.predict(text_vector)[0]
        probabilities = self.ml_model.predict_proba(text_vector)[0]
        
        # Get probability for the predicted class
        class_names = self.ml_model.classes_
        predicted_prob = probabilities[list(class_names).index(prediction)]
        
        return {
            'sentiment': prediction,
            'confidence': predicted_prob,
            'probabilities': dict(zip(class_names, probabilities))
        }
    
    def categorize_feedback(self, text):
        """Categorize feedback into predefined categories"""
        text_lower = text.lower()
        category_scores = {}
        
        for category, keywords in self.category_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            category_scores[category] = score
        
        # Find category with highest score
        best_category = max(category_scores, key=category_scores.get)
        
        # If no keywords match, return 'Other'
        if category_scores[best_category] == 0:
            best_category = 'Other'
        
        return {
            'category': best_category,
            'scores': category_scores
        }
    
    def extract_keywords(self, texts, top_n=10):
        """Extract top keywords using TF-IDF"""
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        tfidf_matrix = tfidf.fit_transform(processed_texts)
        
        # Get feature names and scores
        feature_names = tfidf.get_feature_names_out()
        mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
        
        # Get top keywords
        top_indices = mean_scores.argsort()[-top_n:][::-1]
        top_keywords = [(feature_names[i], mean_scores[i]) for i in top_indices]
        
        return top_keywords
    
    def analyze_comprehensive(self, text):
        """Comprehensive analysis combining all methods"""
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # VADER analysis
        vader_result = self.analyze_sentiment_vader(text)
        
        # TextBlob analysis
        textblob_result = self.analyze_sentiment_textblob(text)
        
        # ML analysis (if model is trained)
        ml_result = None
        if self.ml_model and self.tfidf_vectorizer:
            ml_result = self.analyze_sentiment_ml(text)
        
        # Categorization
        category_result = self.categorize_feedback(text)
        
        # Ensemble sentiment (majority vote)
        sentiments = [vader_result['sentiment'], textblob_result['sentiment']]
        if ml_result:
            sentiments.append(ml_result['sentiment'])
        
        from collections import Counter
        sentiment_counts = Counter(sentiments)
        ensemble_sentiment = sentiment_counts.most_common(1)[0][0]
        
        return {
            'original_text': text,
            'processed_text': processed_text,
            'vader': vader_result,
            'textblob': textblob_result,
            'ml': ml_result,
            'category': category_result,
            'ensemble_sentiment': ensemble_sentiment,
            'confidence': sentiment_counts[ensemble_sentiment] / len(sentiments)
        }
    
    def save_model(self, filepath):
        """Save trained model and vectorizer"""
        model_data = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'ml_model': self.ml_model,
            'category_keywords': self.category_keywords
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        """Load trained model and vectorizer"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.ml_model = model_data['ml_model']
        self.category_keywords = model_data['category_keywords']
