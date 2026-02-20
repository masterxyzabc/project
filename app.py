from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import pandas as pd
import json
import os
from datetime import datetime
from sentiment_analyzer import SentimentAnalyzer
from training_data import training_dataset

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Initialize the sentiment analyzer
analyzer = SentimentAnalyzer()

# Train the ML model on startup
print("Training ML model...")
training_data = pd.DataFrame(training_dataset)
model_metrics = analyzer.train_ml_model(training_data)
print(f"Model trained with accuracy: {model_metrics['accuracy']:.2f}")

# Store analyzed feedback in memory (in production, use a database)
analyzed_feedback = []

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/analyze', methods=['POST'])
def analyze_feedback():
    """Analyze a single feedback"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Please provide feedback text'}), 400
        
        # Perform comprehensive analysis
        result = analyzer.analyze_comprehensive(text)
        
        # Add timestamp
        result['timestamp'] = datetime.now().isoformat()
        result['id'] = len(analyzed_feedback) + 1
        
        # Store in memory
        analyzed_feedback.append(result)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    """Upload and analyze CSV file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Please upload a CSV file'}), 400
        
        # Read CSV
        df = pd.read_csv(file)
        
        # Check if required column exists
        if 'text' not in df.columns and 'feedback' not in df.columns:
            return jsonify({'error': 'CSV must contain a "text" or "feedback" column'}), 400
        
        # Use appropriate column
        text_column = 'text' if 'text' in df.columns else 'feedback'
        
        # Analyze each feedback
        results = []
        for idx, row in df.iterrows():
            text = str(row[text_column]).strip()
            if text:
                result = analyzer.analyze_comprehensive(text)
                result['timestamp'] = datetime.now().isoformat()
                result['id'] = len(analyzed_feedback) + len(results) + 1
                result['row_number'] = idx + 1
                results.append(result)
        
        # Add to analyzed feedback
        analyzed_feedback.extend(results)
        
        return jsonify({
            'message': f'Successfully analyzed {len(results)} feedback entries',
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_analytics')
def get_analytics():
    """Get analytics data for dashboard"""
    try:
        if not analyzed_feedback:
            return jsonify({'error': 'No feedback data available'}), 400
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(analyzed_feedback)
        
        # Sentiment distribution
        sentiment_counts = df['ensemble_sentiment'].value_counts().to_dict()
        
        # Category distribution
        category_counts = df['category'].apply(lambda x: x['category']).value_counts().to_dict()
        
        # Confidence statistics
        confidence_stats = {
            'mean': float(df['confidence'].mean()),
            'min': float(df['confidence'].min()),
            'max': float(df['confidence'].max())
        }
        
        # Timeline data (last 10 entries)
        timeline_data = []
        for item in analyzed_feedback[-10:]:
            timeline_data.append({
                'id': item['id'],
                'sentiment': item['ensemble_sentiment'],
                'category': item['category']['category'],
                'confidence': item['confidence'],
                'timestamp': item['timestamp']
            })
        
        # Extract keywords by sentiment
        positive_texts = [item['original_text'] for item in analyzed_feedback 
                         if item['ensemble_sentiment'] == 'positive']
        negative_texts = [item['original_text'] for item in analyzed_feedback 
                         if item['ensemble_sentiment'] == 'negative']
        
        positive_keywords = analyzer.extract_keywords(positive_texts, 5) if positive_texts else []
        negative_keywords = analyzer.extract_keywords(negative_texts, 5) if negative_texts else []
        
        return jsonify({
            'total_feedback': len(analyzed_feedback),
            'sentiment_distribution': sentiment_counts,
            'category_distribution': category_counts,
            'confidence_stats': confidence_stats,
            'timeline_data': timeline_data,
            'positive_keywords': positive_keywords,
            'negative_keywords': negative_keywords,
            'model_accuracy': model_metrics['accuracy']
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_feedback')
def get_feedback():
    """Get all analyzed feedback"""
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))
        
        # Pagination
        start = (page - 1) * per_page
        end = start + per_page
        
        paginated_feedback = analyzed_feedback[start:end]
        
        return jsonify({
            'feedback': paginated_feedback,
            'total': len(analyzed_feedback),
            'page': page,
            'per_page': per_page,
            'total_pages': (len(analyzed_feedback) + per_page - 1) // per_page
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/export_data')
def export_data():
    """Export analyzed data as CSV"""
    try:
        if not analyzed_feedback:
            return jsonify({'error': 'No data to export'}), 400
        
        # Convert to DataFrame
        export_data = []
        for item in analyzed_feedback:
            export_data.append({
                'id': item['id'],
                'timestamp': item['timestamp'],
                'original_text': item['original_text'],
                'sentiment': item['ensemble_sentiment'],
                'confidence': item['confidence'],
                'category': item['category']['category'],
                'vader_compound': item['vader']['compound'],
                'textblob_polarity': item['textblob']['polarity']
            })
        
        df = pd.DataFrame(export_data)
        
        # Save to CSV
        filename = f'feedback_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(filename, index=False)
        
        return jsonify({
            'message': f'Data exported to {filename}',
            'filename': filename
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear_data', methods=['POST'])
def clear_data():
    """Clear all analyzed data"""
    global analyzed_feedback
    analyzed_feedback = []
    return jsonify({'message': 'All data cleared successfully'})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_trained': analyzer.ml_model is not None,
        'feedback_count': len(analyzed_feedback)
    })

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)