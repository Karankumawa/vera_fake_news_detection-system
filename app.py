# github-karankumawa

from flask import Flask, render_template, request
import re
import nltk
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import logging

app = Flask(__name__, template_folder='./templates', static_folder='./static')

# Set up logging
logging.basicConfig(level=logging.INFO)

# Download NLTK data if not present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

# Load models safely
try:
    with open("model.pkl", 'rb') as f:
        loaded_model = pickle.load(f)
    with open("vector.pkl", 'rb') as f:
        vector = pickle.load(f)
except Exception as e:
    logging.error(f"Error loading models: {e}")
    loaded_model = None
    vector = None

lemmatizer = WordNetLemmatizer()
stpwrds = set(stopwords.words('english'))

def fake_news_det(news):
    if not loaded_model or not vector:
        return -1 # Error state
    try:
        review = re.sub(r'[^a-zA-Z\s]', '', news)
        review = review.lower()
        tokens = nltk.word_tokenize(review)
        
        corpus = [lemmatizer.lemmatize(word) for word in tokens if word not in stpwrds]
        input_data = [' '.join(corpus)]
        
        vectorized_input_data = vector.transform(input_data)
        prediction = loaded_model.predict(vectorized_input_data)
        return prediction[0]
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return -1

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        message = request.form.get('news', '')
        if not message.strip():
             return render_template('prediction.html', prediction_text="Please enter some news text to analyze.", status="error")
        
        pred = fake_news_det(message)
        
        if pred == 1:
            result = "Prediction: The News Looks FAKE ⚠️"
            status = "fake"
        elif pred == 0:
            result = "Prediction: The News Looks REAL ✅"
            status = "real"
        else:
            result = "Prediction could not be completed due to an internal error."
            status = "error"
            
        return render_template("prediction.html", prediction_text=result, status=status, original_text=message)
    
    return render_template('prediction.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)