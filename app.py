from flask import Flask, render_template, request, jsonify
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)

# Load the Naive Bayes model and the CountVectorizer
model = joblib.load('naive_bayes_model.pkl')
cv = joblib.load('count_vectorizer.pkl')  # Make sure you've saved the vectorizer during training

# Define a function to preprocess user input
def clean_text(text):
    text = re.sub('<.*?>', '', text)
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    words = nltk.word_tokenize(text)
    words = [w for w in words if w not in stopwords.words('english')]
    text = ' '.join(words)
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['message']
        cleaned_input = clean_text(user_input)
        input_vector = cv.transform([cleaned_input]).toarray()
        prediction = model.predict(input_vector)
        is_spam = "Spam" if prediction[0] == 1 else "Not Spam"
        return render_template('result.html', is_spam=is_spam)

if __name__ == '__main__':
    app.run(debug=True)


