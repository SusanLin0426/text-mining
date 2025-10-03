# Text Mining

This repository contains homework assignments for the **Text Mining** course, covering classical and modern NLP techniques. Each homework includes Python code and a detailed report.

---

## Homework Overview

### HW1 – TF-IDF & Cosine Similarity
- **Goal**: Compute TF-IDF vectors for a set of documents and calculate cosine similarity between pairs.  
- **Tools**: Python, scikit-learn TF-IDF vectorizer, NLTK stopwords.  
- **Result**: Cosine similarity values show low overlap between randomly chosen docs (e.g., 0.0049).  

### HW2 – Document Classification
- **Goal**: Classify text documents into categories using supervised learning.  
- **Methods**: Bernoulli Naïve Bayes, SVM (Linear & RBF).  
- **Evaluation**: Precision, Recall, F1-score, Precision–Recall curves.  

### HW3 – BERT for Classification
- **Goal**: Use pre-trained **BERT-Tiny (L-2_H-128_A-2)** embeddings with an SVM classifier for text classification.  
- **Special Handling**: Batch processing (batch size = 32) to avoid kernel crashes.  
- **Evaluation**: Precision, Recall, F1-score on held-out testing set.  

---

## Environment
- Python 3.9+
- Libraries:
  - `pandas`, `numpy`
  - `scikit-learn`
  - `nltk`
  - `matplotlib`
  - `transformers`
  - `torch`


