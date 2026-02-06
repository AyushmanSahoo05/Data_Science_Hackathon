# Data_Science_Hackathon
Causal Analysis and Interactive Reasoning over Conversational Data.
# Conversational Transcript Outcome Prediction + Evidence Extraction (Causal Explanation System)

 Project Overview
This project builds an **NLP-based conversational transcript classification system** that predicts the **final outcome/intent** of a conversation (example: *Delivery Investigation, Fraud Alert Investigation, Escalation - Threat of Legal Action*, etc.).

It includes two major approaches:

 Part A: Traditional ML Pipeline (Fast & Lightweight)
- TF-IDF Vectorization
- Linear Support Vector Classifier (LinearSVC)
- Turn-level prediction
- Evidence extraction using decision scores
- Query-based retrieval using cosine similarity

 Part B: Deep Learning Pipeline (Transformer-based)
- Sentence-Transformers (`all-MiniLM-L6-v2`) for embeddings
- Hierarchical attention model (PyTorch)
- Attention-based evidence extraction
- Context memory module for follow-up reasoning
- Evaluation results for ID Recall, Faithfulness and Relevancy Scoring

This system is designed to provide both:
- Accurate predictions
- Interpretability (evidence turns + causal factors)
For Task 1, we built an interpretable classification pipeline that predicts the conversation outcome and extracts the most influential evidence turns responsible for the prediction. For Task 2, we extended the system with a transformer-based embedding model and an attention-based deep learning architecture to generate more context-aware reasoning and evidence extraction.

The repository includes:

Project report explaining the complete methodology and results.

Query Dataset CSV file containing structured query-answer style outputs generated from the system.

A ZIP file containing the full source code and codebase for Task 1 and Task 2 along with trained model(s).

A requirements.txt file listing all required external libraries and their versions to ensure reproducibility of the environment.

This project demonstrates end-to-end implementation including data preprocessing, model training, prediction generation, evidence extraction, evaluation, and output generation
