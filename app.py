import base64
from flask import Flask, request, render_template, send_file, session
import pandas as pd
import pickle
import re
import string
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
import io
import tempfile
import os
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

nltk.download('wordnet')
nltk.download('omw-1.4')

# Load your pre-trained model
with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('vectoriser.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

app = Flask(__name__)
app.static_folder = 'static'  # Set the directory for static files
app.template_folder = 'templates'

# Set the secret key to some random bytes. Keep this really secret!
app.secret_key = os.urandom(24)

STOPWORDS = set([
    'a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
    'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before',
    'being', 'below', 'between', 'both', 'by', 'can', 'd', 'did', 'do',
    'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from',
    'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
    'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
    'into', 'is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
    'me', 'more', 'most', 'my', 'myself', 'now', 'o', 'of', 'on', 'once',
    'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'own', 're', 's', 'same', 'she', "shes", 'should', "shouldve", 'so', 'some', 'such',
    't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
    'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
    'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was',
    'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom',
    'why', 'will', 'with', 'won', 'y', 'you', "youd", "youll", "youre",
    "youve", 'your', 'yours', 'yourself', 'yourselves'
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file and file.filename.endswith('.csv'):
        df = pd.read_csv(file)
        analysis_results = perform_sentiment_analysis(df)

        # Update the DataFrame with sentiment predictions
        df['sentiment'] = [result['sentiment'] for result in analysis_results]

        # Calculate summary statistics
        total_tweets = len(df)
        positive_tweets = sum(df['sentiment'] == 'positive')
        negative_tweets = sum(df['sentiment'] == 'negative')

        # Generate the plot
        plot_data = generate_plot(df)
        
        # Encode the plot data using Base64
        plot_data_base64 = base64.b64encode(plot_data).decode('ascii')

        return render_template('result.html', total_tweets=total_tweets, positive_tweets=positive_tweets,
                               negative_tweets=negative_tweets, plot_data_base64=plot_data_base64)
    else:
        return "Invalid file type"

@app.route('/download')
def download_file():
    output_path = session.get('csv_output_path')
    if output_path and os.path.exists(output_path):
        return send_file(output_path, as_attachment=True, download_name='sentiment_analysis_results.csv')
    else:
        return "No file to download"

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    text = " ".join([word for word in text.split() if word not in STOPWORDS])
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove repeating characters
    text = re.sub(r'(.)\1+', r'\1', text)
    # Remove URLs
    text = re.sub(r'((www.[^s]+)|(https?://[^s]+))', ' ', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Tokenize
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(text)
    # Stemming
    st = PorterStemmer()
    text = [st.stem(word) for word in text]
    # Lemmatizing
    lm = WordNetLemmatizer()
    text = [lm.lemmatize(word) for word in text]
    return " ".join(text)

def perform_sentiment_analysis(df):
    results = []
    for tweet in df['tweet']:
        preprocessed_tweet = preprocess_text(tweet)
        # Transform the tweet using the pre-trained vectorizer
        transformed_tweet = vectorizer.transform([preprocessed_tweet])
        # Predict sentiment
        sentiment = model.predict(transformed_tweet)[0]
        # Convert numeric sentiment to string label
        sentiment_label = 'positive' if sentiment == 1 else 'negative'
        results.append({'tweet': tweet, 'sentiment': sentiment_label})
    return results

def generate_plot(df):
    # Count the occurrences of each sentiment
    sentiment_counts = df['sentiment'].value_counts()
    
    # Create a bar plot
    plt.figure(figsize=(6, 4))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis")
    plt.title('Sentiment Analysis Results')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    
    # Save the plot to a BytesIO object
    plot_buffer = BytesIO()
    plt.savefig(plot_buffer, format='png')
    plot_buffer.seek(0)
    
    # Clear the plot to release memory
    plt.clf()
    plt.close()
    
    return plot_buffer.getvalue()

if __name__ == "__main__":
    app.run(debug=True)
