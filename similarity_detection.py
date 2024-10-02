import streamlit as st
import nltk
import spacy
from textblob import TextBlob
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader
from docx import Document


# Function to calculate Cosine Similarity
def cosine_similarity_func(doc1, doc2):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([doc1, doc2])
    cosine_sim = cosine_similarity(X.toarray())
    return cosine_sim[0][1]


# Function to calculate Jaccard Similarity
def jaccard_similarity(doc1, doc2):
    a = set(doc1.split())
    b = set(doc2.split())
    jaccard_sim = len(a.intersection(b)) / len(a.union(b))
    return jaccard_sim


# Function to read content from uploaded files
def read_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() if page.extract_text() else ""
        return text.strip()
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text.strip()
    elif uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8").strip()
    else:
        st.error("Unsupported file format. Please upload a PDF, Word, or TXT document.")
        return None


# Function to interpret similarity scores
def interpret_similarity(cosine_sim, jac_sim):
    if cosine_sim >= 0.8:
        cos_interpretation = "Highly similar: The resume closely matches the job description."
    elif cosine_sim >= 0.5:
        cos_interpretation = "Moderately similar: The resume has some relevant content."
    else:
        cos_interpretation = "Low similarity: The resume may not align well with the job description."

    if jac_sim >= 0.8:
        jac_interpretation = "Highly similar: The resume closely matches the job description."
    elif jac_sim >= 0.5:
        jac_interpretation = "Moderately similar: The resume has some relevant content."
    else:
        jac_interpretation = "Low similarity: The resume may not align well with the job description."

    return cos_interpretation, jac_interpretation


# Function to analyze dissimilarity reasons
def analyze_dissimilarity(doc1, doc2):
    doc1_set = set(doc1.lower().split())
    doc2_set = set(doc2.lower().split())

    missing_terms_doc1 = doc2_set - doc1_set
    missing_terms_doc2 = doc1_set - doc2_set

    return missing_terms_doc1, missing_terms_doc2


# Function to generate basic analytics
def document_analysis(doc1, doc2):
    doc1_word_count = len(doc1.split())
    doc2_word_count = len(doc2.split())
    return doc1_word_count, doc2_word_count


# Main Streamlit application code
def main():
    st.title('Resume and Job Description Similarity Calculator')

    # User input for 2 documents
    st.header('Upload Documents')
    uploaded_file1 = st.file_uploader('Upload Resume (PDF, Word, TXT):', type=["pdf", "docx", "txt"])
    uploaded_file2 = st.file_uploader('Upload Job Description (PDF, Word, TXT):', type=["pdf", "docx", "txt"])

    if st.button('Calculate Similarity'):
        st.header('Similarity Results')
        if uploaded_file1 is None or uploaded_file2 is None:
            st.warning('Please upload both documents.')
        else:
            # Read the content of the uploaded files
            doc1 = read_file(uploaded_file1)
            doc2 = read_file(uploaded_file2)

            if doc1 is None or doc2 is None:
                return

            # Cosine Similarity
            cos_sim = cosine_similarity_func(doc1, doc2)
            st.write('Cosine Similarity:', cos_sim)

            # Jaccard Similarity
            jac_sim = jaccard_similarity(doc1, doc2)
            st.write('Jaccard Similarity:', jac_sim)

            # Interpret the similarity scores
            cos_interpretation, jac_interpretation = interpret_similarity(cos_sim, jac_sim)
            st.write("### Interpretation")
            st.write("**Cosine Similarity Interpretation:**", cos_interpretation)
            st.write("**Jaccard Similarity Interpretation:**", jac_interpretation)

            # Analyze dissimilarities
            missing_terms_doc1, missing_terms_doc2 = analyze_dissimilarity(doc1, doc2)
            if missing_terms_doc1:
                st.write("### Missing Terms in Resume (Doc1):")
                st.write(", ".join(missing_terms_doc1))
            if missing_terms_doc2:
                st.write("### Missing Terms in Job Description (Doc2):")
                st.write(", ".join(missing_terms_doc2))

            # Generate basic analytics
            doc1_word_count, doc2_word_count = document_analysis(doc1, doc2)
            st.write("### Document Analytics")
            st.write(f"**Word Count of Resume:** {doc1_word_count}")
            st.write(f"**Word Count of Job Description:** {doc2_word_count}")

            # Visualizations
            plt.figure(figsize=(8, 4))
            plt.bar(['Resume', 'Job Description'], [doc1_word_count, doc2_word_count], color=['blue', 'orange'])
            plt.title('Word Count Comparison')
            plt.ylabel('Word Count')
            plt.xlabel('Documents')
            st.pyplot(plt)

            # Estimate chances of getting an interview
            if cos_sim < 0.5 and jac_sim < 0.5:
                st.write("### Chances of Getting an Interview: Low")
                st.write(
                    "The dissimilarity indicates a poor match between your resume and the job description. Consider revising your resume to include more relevant skills and experiences.")
            elif cos_sim < 0.7 or jac_sim < 0.7:
                st.write("### Chances of Getting an Interview: Moderate")
                st.write(
                    "Your resume has some relevant content but lacks certain keywords or skills highlighted in the job description. Adjustments may improve your chances.")
            else:
                st.write("### Chances of Getting an Interview: High")
                st.write(
                    "Your resume closely aligns with the job description, indicating a good match. You have a favorable chance of being invited for an interview.")


if __name__ == '__main__':
    main()
