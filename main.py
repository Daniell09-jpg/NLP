import nltk
import spacy
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

# Download necessary resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# -----------------------
# TEXT PREPROCESSING
# -----------------------
def preprocess(text):
    text = text.lower()  # lowercase
    tokens = word_tokenize(text)  # tokenize
    
    # remove punctuation + stopwords
    clean_tokens = [
        word for word in tokens 
        if word not in string.punctuation and word not in stopwords.words('english')
    ]
    
    # Lemmatization using spaCy
    doc = nlp(" ".join(clean_tokens))
    lemmatized = [token.lemma_ for token in doc]
    
    return lemmatized

# -----------------------
# MAIN NLP FUNCTION
# -----------------------
def analyze_text(text):
    print("\n=== Original Text ===")
    print(text)

    # Preprocess text
    cleaned = preprocess(text)
    print("\n=== Cleaned & Lemmatized Text ===")
    print(cleaned)

    # POS Tagging
    pos_tags = nltk.pos_tag(cleaned)
    print("\n=== Part-of-Speech Tags ===")
    print(pos_tags)

    # NER using spaCy
    doc = nlp(text)
    print("\n=== Named Entities ===")
    for ent in doc.ents:
        print(ent.text, " → ", ent.label_)

    # Sentiment Analysis
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    print("\n=== Sentiment Analysis ===")
    print(sentiment)

# -----------------------
# RUN THE PROGRAM
# -----------------------
if __name__ == "__main__":
    sample = "Python is an amazing programming language created by Guido van Rossum."
    analyze_text(sample)
