# Customer Feedback & Sentiment Analysis Platform

A comprehensive web-based analytical platform that uses advanced NLP techniques to analyze customer reviews, extract insights, and present data through a professional dashboard.

## 🎯 Features

### Core Functionality
- **Multi-method Sentiment Analysis**: Combines VADER rule-based, TextBlob, and ML-based approaches
- **Intelligent Categorization**: Automatically classifies feedback into Product Quality, Delivery, Customer Service, App Experience, Pricing, and Other
- **Keyword Extraction**: Identifies top keywords and themes from positive and negative reviews
- **Bulk Processing**: CSV upload support for analyzing large datasets
- **Real-time Analytics**: Interactive charts and animated visualizations

### Advanced NLP Features
- **Text Preprocessing**: Lowercasing, stopword removal, lemmatization, tokenization
- **Negation Handling**: Properly processes negations like "not good", "not satisfied"
- **Emoji Support**: Converts emojis to text descriptions for analysis
- **Slang Recognition**: Handles common slang and informal language
- **TF-IDF Vectorization**: Advanced feature extraction for ML models

### Dashboard Features
- **Professional UI**: Modern, responsive design with Bootstrap 5
- **Interactive Charts**: Real-time sentiment and category distribution charts
- **Animated Counters**: Smooth number animations for statistics
- **Keyword Clouds**: Visual representation of top positive/negative keywords
- **Feedback Timeline**: Chronological view of analyzed feedback
- **Export Functionality**: Download analyzed data as CSV
- **Pagination**: Efficient handling of large feedback datasets

## 🏗️ Technical Architecture

### Backend Stack
- **Flask**: Web framework for API endpoints
- **Scikit-learn**: Machine learning models and TF-IDF vectorization
- **NLTK**: Natural language processing utilities
- **TextBlob**: Additional sentiment analysis capabilities
- **VADER**: Rule-based sentiment analysis
- **Pandas**: Data manipulation and analysis

### Frontend Stack
- **HTML5/CSS3**: Modern semantic markup and styling
- **Bootstrap 5**: Responsive UI framework
- **Chart.js**: Interactive data visualization
- **JavaScript**: Dynamic client-side functionality
- **Font Awesome**: Professional icon library

### ML Models
- **Logistic Regression**: Primary sentiment classifier
- **TF-IDF Vectorizer**: Feature extraction with n-grams
- **Ensemble Approach**: Combines multiple sentiment analysis methods
- **Custom Training Dataset**: 80 balanced customer reviews

## 📊 Model Performance

The platform uses a custom-trained Logistic Regression model with:
- **Training Dataset**: 80 balanced customer reviews (positive, negative, neutral)
- **Feature Extraction**: TF-IDF with unigrams and bigrams
- **Accuracy**: Typically 85-95% on test data
- **Ensemble Method**: Majority voting across VADER, TextBlob, and ML model

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the project files**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Access the dashboard**:
   Open your browser and navigate to `http://127.0.0.1:5000`

### First Run
The application will automatically:
- Download required NLTK data
- Train the ML model on the built-in dataset
- Display model accuracy in the dashboard header

## 📖 Usage Guide

### Single Feedback Analysis
1. Navigate to the "Single Feedback" tab
2. Enter customer feedback text in the textarea
3. Click "Analyze Feedback" to get instant results
4. View detailed sentiment, category, and confidence scores

### Bulk CSV Upload
1. Switch to the "Bulk Upload (CSV)" tab
2. Prepare a CSV file with a 'text' or 'feedback' column
3. Drag and drop or click to upload the file
4. Wait for processing and view bulk analysis results

### Understanding Results
- **Sentiment Badge**: Shows positive/negative/neutral classification
- **Category Badge**: Indicates the feedback category
- **Confidence Bar**: Visual representation of analysis confidence
- **Keywords**: Top terms extracted from positive/negative feedback

## 📁 Project Structure

```
neuracare/
├── app.py                 # Main Flask application
├── sentiment_analyzer.py  # NLP and ML analysis engine
├── training_data.py       # Training dataset generation
├── requirements.txt       # Python dependencies
├── README.md             # This documentation
├── templates/
│   └── dashboard.html    # Main dashboard UI
└── venv/                 # Virtual environment (if created)
```

## 🔧 API Endpoints

### Core Analysis
- `POST /analyze` - Analyze single feedback text
- `POST /upload_csv` - Upload and analyze CSV file
- `GET /get_analytics` - Get dashboard analytics data
- `GET /get_feedback` - Get paginated feedback list

### Data Management
- `GET /export_data` - Export analyzed data as CSV
- `POST /clear_data` - Clear all analyzed data
- `GET /health` - Health check and model status

## 🎨 Customization

### Adding New Categories
Edit the `_initialize_category_keywords()` method in `sentiment_analyzer.py`:
```python
'New Category': [
    'keyword1', 'keyword2', 'keyword3'
]
```

### Training Custom Models
1. Prepare your training data in the format used in `training_data.py`
2. Update the training dataset
3. Restart the application to retrain the model

### UI Customization
- Modify `templates/dashboard.html` for layout changes
- Update CSS variables for color scheme changes
- Add new charts using Chart.js in the dashboard

## 🔍 Analysis Methods

### VADER Sentiment Analysis
- Rule-based approach optimized for social media text
- Handles emoticons, slang, and capitalization
- Provides compound, positive, negative, and neutral scores

### TextBlob Analysis
- Pattern-based sentiment analysis
- Provides polarity and subjectivity scores
- Good for general text analysis

### ML-Based Analysis
- Logistic Regression classifier
- TF-IDF feature extraction
- Trained on custom customer feedback dataset

### Ensemble Method
- Majority voting across all three methods
- Improves accuracy and reliability
- Provides confidence scores

## 📈 Analytics Features

### Real-time Statistics
- Total feedback count
- Sentiment distribution
- Category breakdown
- Confidence metrics

### Visualizations
- Doughnut chart for sentiment distribution
- Bar chart for category distribution
- Keyword clouds for theme analysis
- Timeline view of feedback trends

### Data Export
- CSV export with all analysis results
- Includes timestamps, sentiment scores, and categories
- Compatible with Excel and data analysis tools

## 🛠️ Troubleshooting

### Common Issues

1. **Model Training Fails**
   - Check internet connection for NLTK downloads
   - Ensure all dependencies are installed
   - Verify Python version compatibility

2. **CSV Upload Issues**
   - Ensure CSV has 'text' or 'feedback' column
   - Check file format and encoding
   - Verify file size limits

3. **Performance Issues**
   - Large CSV files may take time to process
   - Consider processing in batches for very large datasets
   - Monitor memory usage during bulk analysis

### Error Messages
- **"No feedback data available"**: Add some feedback first
- **"Model not trained"**: Restart the application
- **"CSV must contain text column"**: Check CSV file format

## 🚀 Future Enhancements

### Planned Features
- [ ] Real-time feedback streaming
- [ ] Advanced topic modeling (LDA)
- [ ] Multi-language support
- [ ] Integration with customer support platforms
- [ ] Automated report generation
- [ ] API rate limiting and authentication
- [ ] Database persistence
- [ ] Advanced filtering and search

### Performance Improvements
- [ ] Caching for frequently accessed data
- [ ] Background processing for large files
- [ ] Optimized database queries
- [ ] Load balancing for high traffic

## 📄 License

This project is for educational and demonstration purposes. Please ensure compliance with data privacy regulations when using with real customer data.

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve the platform.

## 📞 Support

For technical questions or issues, please refer to the troubleshooting section or create an issue in the project repository.
