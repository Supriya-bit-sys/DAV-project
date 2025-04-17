import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv("resumes.csv")
import nlpaug.augmenter.word as naw

# Initialize Synonym Augmentation
aug = naw.SynonymAug(aug_src='wordnet')

# Apply augmentation to resume text
df['resume_text_aug'] = df['resume_text'].apply(lambda x: aug.augment(x))

# Append new augmented data to dataset
df = pd.concat([df, df[['resume_text_aug', 'job_role']].rename(columns={'resume_text_aug': 'resume_text'})], ignore_index=True)

df = df.drop(columns=['resume_text_aug'])  # Remove temporary column

print("Dataset size after augmentation:", df.shape)  # Should be larger


def preprocess_text(text):
    if isinstance(text, list):  
        text = " ".join(text)  # Convert list to string
    if not isinstance(text, str):
        return ""  # Return empty string if not a valid text
    text = text.lower()  # Convert to lowercase
    return text

# Apply preprocessing
df["clean_resume_text"] = df["resume_text"].apply(preprocess_text)

# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
X = vectorizer.fit_transform(df["clean_resume_text"])  # Use cleaned text
y = df["job_role"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Save model and vectorizer
with open("models/resume_classifier.pkl", "wb") as f:
    pickle.dump(model, f)
with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model trained and saved successfully!")

# Print sample features
print(vectorizer.get_feature_names_out()[:20])  # Top 20 words
