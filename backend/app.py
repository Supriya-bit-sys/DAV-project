from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pdfplumber
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the folder exists
MODEL_PATH = "models/resume_classifier.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

# Load trained model and vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    with pdfplumber.open(pdf_path) as pdf:
        text = " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return text if text else "No text found"

@app.route("/upload", methods=["POST"])
def upload_resume():
    """Handle resume upload and return job role prediction."""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Extract text and predict
    resume_text = extract_text_from_pdf(file_path)
    features = vectorizer.transform([resume_text])
    prediction = model.predict(features)[0]

    return jsonify({"predicted_role": prediction})

if __name__ == "__main__":
    app.run(debug=True)
