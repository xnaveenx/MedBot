import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import pandas as pd
import numpy as np
import re
import Levenshtein
import requests

pytesseract.pytesseract.tesseract_cmd = r'D:\Mini Project\tesseract.exe'
df = pd.read_csv(r'D:\Mini Project\code\cleaned_medicine_dataset_cleaned.csv') 
ignore_terms = ["tablet", "tablets", "mg", "g", "film-coated", "capsule", "500"]
def clean_extracted_text(text):
    text = text.strip()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    for term in ignore_terms:
        text = re.sub(r'\b' + re.escape(term) + r'\b', '', text, flags=re.IGNORECASE)
    return text.strip()
def extract_text_from_image(image_path):
    image = Image.open(image_path)
    image = image.convert("L")
    image = image.filter(ImageFilter.SHARPEN)
    image = image.point(lambda p: p > 150 and 255)
    extracted_text = pytesseract.image_to_string(image)
    return clean_extracted_text(extracted_text)
def closest_match_levenshtein(extracted_text, medical_names):
    cleaned_names = [clean_extracted_text(name) for name in medical_names]
    similarities = [Levenshtein.ratio(extracted_text, name) for name in cleaned_names]
    max_similarity_index = np.argmax(similarities)
    return max_similarity_index, similarities[max_similarity_index]
def query_openai_api(message_content):
    api_key = ''  # Replace with your OpenAI API key
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": message_content}],
        "max_tokens": 100
    }

    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        result = response.json()['choices'][0]['message']['content']
        print(result)
        return result
    else:
        print("Failed to retrieve information from OpenAI.")
        return None


def handle_conversation(medicine_name):
    print(f"You can now ask additional questions about {medicine_name}. Type 'exit' to end the conversation.")
    while True:
        user_query = input("Ask a question: ")
        if user_query.lower() == 'exit':
            print("Ending conversation. Goodbye!")
            break
        else:
            query_openai_api(f"Regarding {medicine_name}, {user_query}")
def main():
    input_type = input("Would you like to provide an image or type the medicine name? (Enter 'image' or 'text'): ").strip().lower()
    
    if input_type == 'image':
        image_path = input("Please enter the path to the image file: ").strip()
        extracted_text = extract_text_from_image(image_path)
        print(f"Extracted Text: {extracted_text}")
        
    elif input_type == 'text':
        extracted_text = input("Please type the name of the medicine: ").strip()
        extracted_text = clean_extracted_text(extracted_text)
        print(f"Processed Text: {extracted_text}")
        
    else:
        print("Invalid input type. Please enter 'image' or 'text'.")
        return

    index, similarity_score = closest_match_levenshtein(extracted_text, df['Medicine Name'].values)

    if similarity_score > 0.5:
        medicine_name = df.iloc[index]['Medicine Name']
        print(f"Medicine Found: {medicine_name}")
        print(df.iloc[index])
        print(f"Similarity Score: {similarity_score:.2f}")

        handle_conversation(medicine_name)
        
    else:
        print("No close match found. Searching online...")
        result = query_openai_api(f"Please provide information about the medicine '{extracted_text}' including its uses, side effects, and composition.")
        
        if result:
            print("Online Search Result:")
            print(result)
            handle_conversation(extracted_text)

if __name__ == "__main__":
    main()
