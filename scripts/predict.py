import pickle
import pdfplumber
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Load the trained model and vectorizer
with open("models/resume_classifier.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Function to clean resume text
def clean_resume_text(text):
    text = re.sub(r'\W+', ' ', text)
    words = word_tokenize(text.lower())
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Extract text from a new resume
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return clean_resume_text(text)

# Predict job role for a new resume
def predict_job_role(pdf_path):
    extracted_text = extract_text_from_pdf(pdf_path)
    features = vectorizer.transform([extracted_text])
    predicted_role = model.predict(features)[0]
    return predicted_role

# Test with a new resume
pdf_file = "resumes/new_resume.pdf"  # Change this file name
predicted_role = predict_job_role(pdf_file)
print(f"Predicted Job Role: {predicted_role}")
