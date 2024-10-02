# Resume and Job Description Similarity Calculator

This application allows users to analyze the similarity between a resume and a job description. By using various similarity metrics, users can gain insights into how well their resume aligns with job requirements, helping them improve their chances of landing an interview.

## Table of Contents
- [Introduction](#introduction)
- [Algorithms Used](#algorithms-used)
- [Code Explanation](#code-explanation)
- [How to Use](#how-to-use)
- [Use Cases](#use-cases)
- [Conclusion](#conclusion)

## Introduction

In the competitive job market, ensuring that your resume aligns with job descriptions is crucial. This application evaluates the similarity between two text documents: a resume and a job description. It provides insights into the content of the documents and helps users identify areas for improvement.

## Algorithms Used

1. **Cosine Similarity**:
   - This metric measures the cosine of the angle between two non-zero vectors in a multi-dimensional space. It calculates how similar two documents are irrespective of their size.
   - Formula: 
     \[
     \text{Cosine Similarity} = \frac{A \cdot B}{\|A\| \|B\|}
     \]
   - A value of 1 indicates perfect similarity, while a value of 0 indicates no similarity.

2. **Jaccard Similarity**:
   - This metric evaluates the similarity between two sets by dividing the size of their intersection by the size of their union.
   - Formula:
     \[
     \text{Jaccard Similarity} = \frac{|A \cap B|}{|A \cup B|}
     \]
   - A value of 1 indicates complete overlap, while a value of 0 indicates no overlap.

## Code Explanation

The application is built using Python's Streamlit library and includes the following components:

- **File Upload**: Users can upload a resume and a job description in PDF, Word, or TXT formats.
- **Text Processing**: The uploaded documents are read, and their content is extracted for analysis.
- **Similarity Calculation**: Both cosine and Jaccard similarity metrics are computed for the two documents.
- **Interpretation of Results**: The application provides interpretations based on the similarity scores and highlights missing terms from each document.
- **Visual Analytics**: It generates a bar chart comparing the word count of both documents.

### Key Functions

- `read_file(uploaded_file)`: Reads the content from the uploaded files based on their format.
- `cosine_similarity_func(doc1, doc2)`: Calculates cosine similarity between two documents.
- `jaccard_similarity(doc1, doc2)`: Calculates Jaccard similarity between two documents.
- `interpret_similarity(cosine_sim, jac_sim)`: Interprets the similarity scores and provides feedback.
- `analyze_dissimilarity(doc1, doc2)`: Identifies missing terms in both documents to highlight areas for improvement.
- `document_analysis(doc1, doc2)`: Analyzes the word count of the documents.

## How to Use

1. Install the required libraries:
   ```bash
   pip install streamlit nltk spacy textblob sklearn matplotlib PyPDF2 python-docx
