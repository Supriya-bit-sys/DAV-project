import os
import pdfplumber
import re
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Define the folder where resumes are stored
resume_folder = "resumes"
output_csv = "resumes.csv"

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    with pdfplumber.open(pdf_path) as pdf:
        text = " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return text.strip()

def clean_resume_text(text):
    """Cleans extracted resume text."""
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    words = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(words)

# Process all resumes in the folder
data = []
for filename in os.listdir(resume_folder):
    if filename.endswith(".pdf"):
        file_path = os.path.join(resume_folder, filename)
        extracted_text = extract_text_from_pdf(file_path)
        cleaned_text = clean_resume_text(extracted_text)
        data.append({"filename": filename, "resume_text": cleaned_text, "job_role": None})  # Placeholder for job role


# Convert to DataFrame and save as CSV
df = pd.DataFrame(data)
df.to_csv(output_csv, index=False)

print(f"Extracted and cleaned text from {len(data)} resumes and saved to {output_csv}")
