from flask import Flask, render_template, request
import pytesseract
from PIL import Image, ImageFilter
import pandas as pd
import numpy as np
import re
import Levenshtein

app = Flask(__name__)

# Set Tesseract and CSV path
pytesseract.pytesseract.tesseract_cmd = r'D:\Mini Project\tesseract.exe'
df = pd.read_csv('cleaned_medicine_dataset_cleaned.csv')

ignore_terms = ["tablet", "tablets", "mg", "g", "film-coated", "capsule", "500"]

def clean_extracted_text(text):
    text = text.strip()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    for term in ignore_terms:
        text = re.sub(r'\b' + re.escape(term) + r'\b', '', text, flags=re.IGNORECASE)
    return text.strip()

def extract_text_from_image(image):
    image = image.convert("L")
    image = image.filter(ImageFilter.SHARPEN)
    image = image.point(lambda p: p > 150 and 255)
    extracted_text = pytesseract.image_to_string(image)
    return clean_extracted_text(extracted_text)

def closest_match_levenshtein(extracted_text, medical_names):
    cleaned_names = [clean_extracted_text(name) for name in medical_names]
    similarities = [Levenshtein.ratio(extracted_text, name) for name in cleaned_names]
    max_index = np.argmax(similarities)
    return max_index, similarities[max_index]

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None

    if request.method == 'POST':
        text_input = request.form.get('medicine_text')
        image_file = request.files.get('medicine_image')

        if image_file and image_file.filename != '':
            image = Image.open(image_file.stream)
            extracted_text = extract_text_from_image(image)
        elif text_input:
            extracted_text = clean_extracted_text(text_input)
        else:
            extracted_text = ""

        if extracted_text:
            index, score = closest_match_levenshtein(extracted_text, df['Medicine Name'].values)

            if score > 0.5:
                medicine = df.iloc[index]
                result = {
                    "source": "local",
                    "name": medicine['Medicine Name'],
                    "details": medicine.to_dict(),
                    "score": f"{score:.2f}"
                }
            else:
                result = {
                    "source": "none",
                    "name": extracted_text,
                    "details": "No close match found in local dataset.",
                    "score": f"{score:.2f}"
                }

    return render_template("index.html", result=result)

if __name__ == '__main__':
    app.run(debug=True)
