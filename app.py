import newspaper
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sqlite3
import requests
from bs4 import BeautifulSoup
from rake_nltk import Rake
import nltk
nltk.download(['stopwords', 'punkt'], download_dir='C__/Users/ss/Desktop/Sentiment-Analysis/myenv/nltk_data')

app = Flask(__name__)

# Load the saved Keras model
model = load_model('Model/sentiment_analysis_model.h5')

# Load or initialize your tokenizer
tokenizer = Tokenizer()
max_length = 100

# Initialize SQLite database
conn = sqlite3.connect('./Database/analyzed_results.db', check_same_thread=False)
c = conn.cursor()

# Create table if not exists
c.execute('''CREATE TABLE IF NOT EXISTS analyzed_text (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, sentiment TEXT, keywords TEXT)''')
conn.commit()

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Analyze route
@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        link = request.form['input_text']
        input_text, keywords = get_news(link)
        print(f"These are the keywords: {keywords}")

        # Define sentiment thresholds (modify these values if needed)
        POSITIVE_THRESHOLD = 0.5
        NEGATIVE_THRESHOLD = 1 - POSITIVE_THRESHOLD  # Threshold for negative sentiment

        # Load the model
        model = load_model('./Model/sentiment_analysis_model.h5')

        # Tokenize the text
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts([input_text])
        text_sequence = tokenizer.texts_to_sequences([input_text])

        # Pad the sequence
        MAX_SEQUENCE_LENGTH = 30
        text_sequence = pad_sequences(text_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

        # Predict sentiment
        prediction_probs = model.predict(text_sequence)
        prediction = 'positive' if prediction_probs[0][0] > POSITIVE_THRESHOLD else (
            'negative' if prediction_probs[0][0] < NEGATIVE_THRESHOLD else 'neutral')

        # Save the analyzed text, sentiment, and keywords to the database
        # keyword_str = ', '.join([f"{keyword} ({score})" for keyword, score in keywords])
        keyword_str = ', '.join(keywords)
        c.execute('''INSERT INTO analyzed_text (text, sentiment, keywords) VALUES (?, ?, ?)''',
                  (input_text, prediction, keyword_str))
        conn.commit()

        return redirect(url_for('result'))

# Result route
@app.route('/result')
def result():
    # Fetch previously analyzed results from the database
    c.execute('''SELECT * FROM analyzed_text''')
    analyzed_results = c.fetchall()
    return render_template('result.html', analyzed_results=analyzed_results)

# Fetch news from link and extract keywords
def get_news(link):
    # Fetch the website content
    response = requests.get(link)
    html_content = response.content

    # Parse the HTML content
    soup = BeautifulSoup(html_content, "html.parser")

    # Extract the text content
    text_content = soup.get_text()

    # Extract keywords from the text content
    rake = Rake()
    rake.extract_keywords_from_text(text_content)
    keywords = rake.get_ranked_phrases()

    return text_content, keywords

# Preprocess function
def preprocess(text):
    return text.lower()

if __name__ == '__main__':
    app.run(debug=True)