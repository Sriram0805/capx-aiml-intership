
# Stock Movement Prediction Using Social Media Scraping

This project develops a machine learning model that predicts stock movements by scraping user-generated content from social media platforms (e.g., Twitter). The model analyzes sentiment from stock-related discussions and uses this to forecast stock price trends.

## Objective

- Scrape relevant data from social media (Twitter) focused on stock market discussions.
- Perform sentiment analysis on scraped content to predict stock price movements.
- Build a machine learning model to predict stock movements based on sentiment and other features.

## Features

- **Data Scraping**: Scrape real-time stock-related tweets using the Twitter API.
- **Sentiment Analysis**: Perform sentiment analysis on tweets using TextBlob.
- **Machine Learning**: Build a machine learning model (Logistic Regression) to predict stock movement based on tweet sentiment.
- **Model Evaluation**: Measure accuracy, precision, and recall of the prediction model.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Project Structure](#project-structure)
4. [Data Scraping](#data-scraping)
5. [Model Training](#model-training)
6. [Evaluation](#evaluation)
7. [Future Enhancements](#future-enhancements)
8. [License](#license)

## Installation

### Prerequisites

- Python 3.x
- Twitter Developer Account for API access
- Libraries: `tweepy`, `textblob`, `pandas`, `scikit-learn`, `nltk`

### Installing Dependencies

```bash
pip install tweepy textblob pandas scikit-learn nltk
```

## Usage

### Step 1: Set Up Twitter API Credentials

Create a Twitter Developer account and generate the API keys and tokens. Add your credentials in the `scrape_tweets` function inside the code.

```python
# Twitter API credentials
api_key = 'YOUR_API_KEY'
api_secret_key = 'YOUR_API_SECRET_KEY'
access_token = 'YOUR_ACCESS_TOKEN'
access_token_secret = 'YOUR_ACCESS_TOKEN_SECRET'
```

### Step 2: Scrape Tweets

Use the following function to scrape tweets containing stock-related keywords (e.g., Tesla) and perform sentiment analysis.

```python
tweets_df = scrape_tweets('Tesla', count=200)
```

### Step 3: Preprocess Data

The tweets are cleaned, and text is transformed into numerical data using the Bag of Words model for input into the machine learning model.

### Step 4: Train the Model

The Logistic Regression model is trained on the preprocessed tweet data:

```python
model.fit(X_train, y_train)
```

### Step 5: Evaluate the Model

Evaluate the model using accuracy, precision, and recall metrics:

```python
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
```

## Project Structure

```bash
├── README.md
├── stock_movement_prediction.py  # Main Python script for scraping and model training
├── requirements.txt              # Python dependencies
└── data/                         # Directory to store scraped data
```

## Data Scraping

We use Tweepy to scrape tweets in real-time and perform sentiment analysis using TextBlob. The tweets are cleaned to remove unwanted characters like URLs, special characters, and stop words.

## Model Training

The model uses a Bag of Words approach to vectorize tweet text and trains a Logistic Regression model to predict stock price movement (positive or negative) based on tweet sentiment.

## Evaluation

The model is evaluated using:

- **Accuracy**: Percentage of correct predictions.
- **Precision**: Number of true positive results divided by the number of positive results predicted by the model.
- **Recall**: Number of true positive results divided by the number of actual positive results.

## Future Enhancements

1. **Data Source Expansion**: Scrape data from additional platforms such as Reddit or Telegram.
2. **Feature Engineering**: Incorporate additional features like frequency of mentions, topic modeling, or sentiment context.
3. **Advanced Models**: Use more sophisticated machine learning models such as LSTM or Transformers for better prediction accuracy.
4. **Real-time Predictions**: Deploy the model for real-time stock movement predictions based on live social media data.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
