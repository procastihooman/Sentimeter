from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from flask import Flask, render_template,request
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
import pandas as pd
import string

feedback_data = pd.read_csv('Feedback.csv', names=['FEEDBACK', 'ANALYSIS']) 

feedback = feedback_data['FEEDBACK'].values
analysis = feedback_data['ANALYSIS'].values
feedback_train, feedback_test, y_train, y_test = train_test_split(feedback, analysis, test_size=0.2, random_state=1000)

punctuations = string.punctuation
parser = English()
stopwords = list(STOP_WORDS)
def spacy_tokenizer(utterance):
    tokens = parser(utterance)
    return [token.lemma_.lower().strip() for token in tokens if token.text.lower().strip() not in stopwords and token.text not in punctuations]

vectorizer = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))
vectorizer.fit(feedback_train)

X_train = vectorizer.transform(feedback_train)
X_test = vectorizer.transform(feedback_test)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('text.html')

@app.route("/predict",methods=['POST'])
def predict():
    reviews = request.form.values()
    reply = vectorizer.transform(reviews)
    prediction = classifier.predict(reply)
    if prediction == ['1']:
        return render_template('text.html', reply="We are so happy that you liked our product")
    else:
        return render_template('text.html', reply="Sorry, for the disappointment caused by our product. Would you like me to call the customer care?")


if __name__ == "__main__":
    app.run(debug=True)