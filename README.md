# Veritas – Fake News Detection System

Repository: Veritas-FakeNewsDetection

## Project Overview
This project develops a machine learning model to classify news articles as real or fake using Natural Language Processing (NLP).

The analysis focuses on transforming unstructured news headlines into numerical features using TF-IDF and training a Random Forest classification model to support scalable misinformation detection.

## Dataset Overview
- Approximately 9.6K news articles
- Text-based news headline data
- Target variable: Real or Fake news label
- Feature representation: TF-IDF with 5,000 features
- Additional features from news source and news link information

## Analytical Objective
The objective of this project is to detect misleading news content by identifying textual patterns and source-based signals that help distinguish real news from fake news.

## Key Challenges
The raw dataset and classification task contained several analytical challenges:

- Unstructured text data with high dimensionality
- Subtle differences between real and fake news content
- Noise and variability in writing style and language
- Limited contextual understanding when using TF-IDF features
- Difficulty in detecting misleading content that mimics legitimate news

## Workflow
1. Data Loading and Initial Exploration
2. Data Cleaning and Label Filtering
3. Text Preprocessing
4. TF-IDF Feature Extraction
5. Source-Based Feature Engineering
6. Random Forest Model Training
7. Model Evaluation with Accuracy, F1-score, and Confusion Matrix
8. Error Analysis and Improvement Planning

## Tools & Technologies
- Python
- pandas, matplotlib, seaborn
- scikit-learn
- TF-IDF Vectorization
- Random Forest Classifier
- Classification Metrics
- Confusion Matrix Analysis

## Model Performance Summary
| Metric | Result |
|---|---:|
| Total Articles | ~9.6K |
| TF-IDF Features | 5,000 |
| Model | Random Forest |
| Accuracy | 72.8% |
| F1-score | 0.72 |

## Key Insights
- The model achieved an accuracy of 72.8% and an F1-score of 0.72.
- The confusion matrix showed stronger performance in identifying real news compared to fake news.
- Fake news detection remains challenging because misleading content can mimic legitimate writing patterns.
- TF-IDF was effective for converting text into structured numerical features, but it may not fully capture deeper contextual meaning.
- Source-based features helped enhance the model beyond headline text alone.

## Future Improvements
- Compare additional models such as Logistic Regression, Naive Bayes, XGBoost, and BERT.
- Improve text preprocessing and feature engineering.
- Conduct deeper error analysis on misclassified fake news.
- Explore contextual embeddings to improve language understanding.
- Deploy the model into a simple web application for real-world usability.

## Project Structure

```text
Veritas-FakeNewsDetection/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   └── fake_or_real.py
│
├── reports/
│   └── fake_news_detection_notebook.html
│
├── data/
│   └── README.md
│
└── images/
    ├── confusion_matrix.png
    ├── tfidf_heatmap.png
    ├── label_distribution.png
    ├── top_sources.png
    └── feature_importance.png
