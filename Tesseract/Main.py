!pip install pytesseract
!sudo apt-get install tesseract-ocr
!brew install tesseract
!pip install pytesseract
!pip install python-Levenshtein
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten, Bidirectional, LSTM
import pytesseract
from PIL import Image
import re
from difflib import get_close_matches
import Levenshtein

# Load the dataset
df = pd.read_csv('/content/cleaned_medicine_dataset.csv')

# Prepare data
medical_names = df['Composition'].values

# Tokenize the medical names
tokenizer = Tokenizer()
tokenizer.fit_on_texts(medical_names)
sequences = tokenizer.texts_to_sequences(medical_names)
word_index = tokenizer.word_index

# Pad sequences to ensure uniform length
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length)

# Define the neural network model
model = Sequential([
    Embedding(input_dim=len(word_index) + 1, output_dim=100, input_length=max_length),
    Bidirectional(LSTM(64)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(len(medical_names), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model (for simplicity, we're using medical_names as both input and target)
model.fit(padded_sequences, np.arange(len(medical_names)), epochs=20)

# Function to clean and normalize the extracted text
def clean_extracted_text(text):
    text = text.strip()  # Remove leading and trailing whitespace
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove unwanted characters
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.lower()

# Function to extract text from an image using Tesseract OCR
def extract_text_from_image(image_path):
    image = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(image)
    return clean_extracted_text(extracted_text)

# Function to find the closest match using Levenshtein distance
def closest_match_levenshtein(extracted_text, medical_names):
    similarities = [Levenshtein.ratio(extracted_text, name) for name in medical_names]
    max_similarity_index = np.argmax(similarities)
    return max_similarity_index

# Function to predict the most similar medical names
def predict_similar_names(input_name, model, tokenizer, max_length, df):
    input_seq = tokenizer.texts_to_sequences([input_name])
    padded_input_seq = pad_sequences(input_seq, maxlen=max_length)

    # Predict the closest match using the model
    predictions = model.predict(padded_input_seq)
    predicted_index = np.argmax(predictions[0])

    # Fuzzy match or Levenshtein as a fallback
    levenshtein_index = closest_match_levenshtein(input_name, df['Composition'].values)

    # Return the row with the highest confidence or closest match
    return df.iloc[levenshtein_index] if predictions[0][predicted_index] < 0.6 else df.iloc[predicted_index]
# Example: Extract text from an image
image_path = '/content/paracetamol.jpeg'
extracted_text = extract_text_from_image(image_path)

print(f"Extracted Text: {extracted_text}")

# Use the extracted text as input to the model
row_details = predict_similar_names(extracted_text, model, tokenizer, max_length, df)

print("Medicine Found")
print(row_details)