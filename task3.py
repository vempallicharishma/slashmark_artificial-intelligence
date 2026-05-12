import os
import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Folder containing documents
DATA_FOLDER = "."

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))

    filtered_tokens = [
        word for word in tokens
        if word.isalnum() and word not in stop_words
    ]

    return " ".join(filtered_tokens)

# Read documents
documents = []
file_names = []

for file in os.listdir(DATA_FOLDER): 
    if file.endswith(".txt"):
      file_path = os.path.join(DATA_FOLDER, file)

      with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        print(text)
        processed_text = preprocess_text(text)

        documents.append(processed_text)
        file_names.append(file)

# Convert text into TF-IDF vectors
vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(documents)

# Compute similarity matrix
similarity_matrix = cosine_similarity(tfidf_matrix)

# Display similarity results
results = []

for i in range(len(file_names)):
    for j in range(i + 1, len(file_names)):

        similarity_score = similarity_matrix[i][j] * 100

        results.append({
            "Document 1": file_names[i],
            "Document 2": file_names[j],
            "Similarity (%)": round(similarity_score, 2)
        })

# Create DataFrame
df = pd.DataFrame(results)

print("\nPlagiarism Detection Results:\n")
print(df)

# Save results
df.to_csv("plagiarism_report.csv", index=False)

print("\nReport saved as plagiarism_report.csv")