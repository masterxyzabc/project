# Training dataset for sentiment analysis
# Balanced dataset with positive, negative, and neutral reviews

training_dataset = {
    'text': [
        # Positive Reviews (20 examples)
        "The product quality is amazing and delivery was fast.",
        "Excellent customer service, very satisfied!",
        "Great experience, will buy again.",
        "Loved the packaging and the support team was helpful.",
        "The app is smooth and easy to use.",
        "Outstanding quality and fantastic customer support.",
        "Best purchase I've made this year, highly recommended!",
        "Quick delivery and the product exceeded my expectations.",
        "The interface is intuitive and the features are amazing.",
        "Customer service went above and beyond to help me.",
        "Product works exactly as described, very happy with it.",
        "Fast shipping and great value for money.",
        "The team is responsive and the quality is top-notch.",
        "Absolutely love this product, worth every penny!",
        "Smooth user experience and excellent design.",
        "Great customer service and product quality.",
        "Delivery was on time and the product is perfect.",
        "Very impressed with the quality and service.",
        "The app works flawlessly, no issues at all.",
        "Outstanding experience from start to finish.",
        
        # Negative Reviews (20 examples)
        "Very disappointed with the product.",
        "Delivery was late and customer service was rude.",
        "The product stopped working after two days.",
        "Terrible experience, waste of money.",
        "App crashes frequently and is slow.",
        "Poor quality and terrible customer support.",
        "Product arrived broken and no one responds to emails.",
        "Worst purchase ever, completely useless.",
        "The app is full of bugs and crashes constantly.",
        "Customer service is unhelpful and rude.",
        "Product quality is very poor, not worth the price.",
        "Delivery took forever and the item was damaged.",
        "The app never works properly, very frustrating.",
        "Completely dissatisfied with the service.",
        "Product broke after one week of use.",
        "Terrible experience, would not recommend.",
        "The interface is confusing and the app is slow.",
        "Poor quality control and bad customer service.",
        "Waste of money, product doesn't work as advertised.",
        "Very disappointed with the entire experience.",
        
        # Neutral Reviews (20 examples)
        "The product is okay.",
        "Average experience.",
        "It works as expected.",
        "Nothing special but not bad either.",
        "The delivery time was standard.",
        "Product meets basic requirements.",
        "It's fine, nothing extraordinary.",
        "Decent quality for the price.",
        "The app does what it's supposed to do.",
        "Standard service, no complaints.",
        "Product is functional but could be better.",
        "Average customer service experience.",
        "It works, nothing more to say.",
        "The quality is acceptable.",
        "Normal delivery time.",
        "The app is usable but has room for improvement.",
        "Product is as described, nothing more.",
        "Service was okay, could be better.",
        "It meets expectations but doesn't exceed them.",
        "Standard product, average quality.",
        
        # Additional mixed reviews for better training (20 examples)
        "Good product but delivery was slow.",
        "Great quality but customer service needs improvement.",
        "App works well but interface could be better.",
        "Fast delivery but product quality is average.",
        "Customer service is helpful but product is overpriced.",
        "The app has great features but crashes sometimes.",
        "Product quality is excellent but packaging was poor.",
        "Delivery was quick but item arrived damaged.",
        "Great customer service but the product didn't work.",
        "The app is user-friendly but lacks some features.",
        "Product is good value but takes too long to deliver.",
        "Customer service responds quickly but isn't always helpful.",
        "The app works well but the design is outdated.",
        "Quality is good but the price is too high.",
        "Fast delivery but the product doesn't match the description.",
        "Great features but the app is too complicated.",
        "Product works well but customer service is slow.",
        "The app is stable but lacks important features.",
        "Good quality but the shipping cost is too high.",
        "Customer service is friendly but not very knowledgeable."
    ],
    'sentiment': [
        # Positive labels
        'positive', 'positive', 'positive', 'positive', 'positive',
        'positive', 'positive', 'positive', 'positive', 'positive',
        'positive', 'positive', 'positive', 'positive', 'positive',
        'positive', 'positive', 'positive', 'positive', 'positive',
        
        # Negative labels
        'negative', 'negative', 'negative', 'negative', 'negative',
        'negative', 'negative', 'negative', 'negative', 'negative',
        'negative', 'negative', 'negative', 'negative', 'negative',
        'negative', 'negative', 'negative', 'negative', 'negative',
        
        # Neutral labels
        'neutral', 'neutral', 'neutral', 'neutral', 'neutral',
        'neutral', 'neutral', 'neutral', 'neutral', 'neutral',
        'neutral', 'neutral', 'neutral', 'neutral', 'neutral',
        'neutral', 'neutral', 'neutral', 'neutral', 'neutral',
        
        # Mixed labels (leaning towards neutral)
        'neutral', 'neutral', 'neutral', 'neutral', 'neutral',
        'neutral', 'neutral', 'neutral', 'neutral', 'neutral',
        'neutral', 'neutral', 'neutral', 'neutral', 'neutral',
        'neutral', 'neutral', 'neutral', 'neutral', 'neutral'
    ]
}

# Additional test data for validation
test_data = {
    'text': [
        "This product is absolutely fantastic! Love it!",
        "Terrible quality and the worst customer service ever.",
        "It's okay, does what it needs to do.",
        "The app is amazing but crashes occasionally.",
        "Delivery was super fast and the product is perfect!",
        "Completely disappointed, waste of time and money.",
        "Average product, nothing special about it.",
        "Great features but the price is too high."
    ],
    'sentiment': [
        'positive', 'negative', 'neutral', 'neutral', 
        'positive', 'negative', 'neutral', 'neutral'
    ]
}

if __name__ == "__main__":
    import pandas as pd
    
    # Create DataFrames
    train_df = pd.DataFrame(training_dataset)
    test_df = pd.DataFrame(test_data)
    
    print("Training Dataset:")
    print(f"Total samples: {len(train_df)}")
    print(f"Positive: {len(train_df[train_df['sentiment'] == 'positive'])}")
    print(f"Negative: {len(train_df[train_df['sentiment'] == 'negative'])}")
    print(f"Neutral: {len(train_df[train_df['sentiment'] == 'neutral'])}")
    
    print("\nTest Dataset:")
    print(f"Total samples: {len(test_df)}")
    
    # Save to CSV for later use
    train_df.to_csv('training_data.csv', index=False)
    test_df.to_csv('test_data.csv', index=False)
    print("\nDatasets saved to CSV files.")
