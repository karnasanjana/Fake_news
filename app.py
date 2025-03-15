import streamlit as st
import nltk
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.neural_network import MLPClassifier  # Assuming you're using MLPClassifier, adjust if you're using a Keras model

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize the lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Define the preprocessing function
def preprocess_text(text):
    # Tokenize the text into words
    tokens = text.split()

    # Remove stopwords, lemmatize, and keep only alphabetic words
    cleaned_tokens = [
        lemmatizer.lemmatize(token.lower())  # Lemmatize and convert to lowercase
        for token in tokens if token.lower() not in stop_words and token.isalpha()
    ]

    # Return the preprocessed tokens as a single string
    return ' '.join(cleaned_tokens)

# Load the pre-trained TF-IDF vectorizer and MLP model
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
mlp_model = joblib.load('mlp_model.pkl')  # Assuming you have the trained MLP model saved

# Streamlit UI Setup
st.title('Fake News Detection with MLP Model')

st.write("""
    This app classifies news articles as **Real** or **Fake** based on the input text using an MLP model.
    You can enter a sentence below to get a prediction.
""")

# User input: Enter text to classify
user_input = st.text_area("Enter the news article text here:")

# Preprocess and predict when user clicks the button
if st.button("Predict"):
    if user_input:
        # Clean and preprocess the input text
        cleaned_text = preprocess_text(user_input)

        # Transform the input text using the trained TF-IDF vectorizer
        input_tfidf = tfidf_vectorizer.transform([cleaned_text])

        # Make predictions using the MLP model
        mlp_prediction = mlp_model.predict(input_tfidf)

        # Convert MLP predictions to True/False (or 'Real'/'Fake')
        prediction_label = "Real" if mlp_prediction[0] == 1 else "Fake"

        # Display the prediction
        st.write(f"**Prediction**: {prediction_label}")
    else:
        st.error("Please enter some text to classify.")
