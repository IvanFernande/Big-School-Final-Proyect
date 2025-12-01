# Project 1 -- Predictive Modeling & Business Metrics

## Priority Classification of IT Support Tickets with SLA Impact Analysis

This project focuses on building an end-to-end Machine Learning system
that predicts the priority level of IT support tickets using Natural
Language Processing (NLP) and metadata. The business goal is to reduce
SLA violations, optimize support response times, and quantify
operational and economic impact.

------------------------------------------------------------------------

## 0. Project Overview

### 🎯 Business Objective

Automatically classify support tickets into **High**, **Medium**, or
**Low** priority to help support teams react faster, prevent SLA
breaches, and improve efficiency.

### 📂 Dataset

The dataset contains real-world IT support tickets with the following
main fields:

-   `Body` -- Full text of the ticket\
-   `Department` -- Reporting area\
-   `Priority` -- Target variable (low, medium, high)\
-   `Tags` -- Ticket metadata

------------------------------------------------------------------------

## 1. Data Understanding & Initial Diagnostics

### 1.1 Dataset Loading

Load the CSV file and inspect columns, types, and shape.

### 1.2 Exploratory Data Analysis (EDA)

-   Distribution of priority classes\
-   Length of ticket descriptions\
-   Tag frequency\
-   Department distribution\
-   Word frequency analysis

**Visualizations:** - Histograms\
- Countplots\
- Wordclouds by priority\
- Boxplots for text length

------------------------------------------------------------------------

## 2. Data Cleaning & Preprocessing

### 2.1 Text Cleaning

-   Lowercasing\
-   Stopword removal\
-   Tokenization\
-   Stemming/Lemmatization\
-   Removing special characters

### 2.2 Metadata Cleaning

-   Normalize departments\
-   Parse tags lists\
-   Remove unnecessary index columns

### 2.3 Outlier Handling

-   Extremely short/long ticket descriptions

------------------------------------------------------------------------

## 3. Feature Engineering (Core Value Section)

### 3.1 Text Features

-   TF-IDF vectors\
-   Bag-of-Words\
-   N-grams\
-   Word embeddings (GloVe, Word2Vec)\
-   DistilBERT embeddings (optional, adds strong value)

### 3.2 Derived Numerical Features

-   Ticket length (characters/tokens)\
-   Number of tags\
-   Sentiment score\
-   Presence of critical keywords (urgent, outage, failure)\
-   Lexical complexity

### 3.3 Categorical Features

-   One-hot encode `Department`\
-   Multi-hot encode `Tags`

------------------------------------------------------------------------

## 4. Dataset Enhancement (Optional but Highly Recommended)

### 4.1 SLA Expected Time

Define expected resolution time based on priority: - High → 4h\
- Medium → 12h\
- Low → 48h

### 4.2 Simulated "Actual Resolution Time"

Generate realistic resolution times to evaluate business impact.

### 4.3 SLA Violation Feature

    SLA_violated = actual_resolution_time > SLA_expected_time

This enables: - Business metrics\
- Cost simulation\
- Scenario analysis

------------------------------------------------------------------------

## 5. Modeling

### 5.1 Baseline Models

-   Logistic Regression\
-   Naive Bayes\
-   Support Vector Machines\
-   Random Forest

### 5.2 Advanced Models

-   XGBoost\
-   LightGBM

### 5.3 State-of-the-Art NLP (Optional)

-   DistilBERT fine-tuning\
-   BERT embeddings + tabular model

### 5.4 Validation Strategy

-   Train/test split\
-   Cross-validation\
-   GridSearch / RandomSearch

### 5.5 Technical Metrics

-   F1-score (macro and per class)\
-   Accuracy\
-   Confusion Matrix\
-   ROC-AUC (One-vs-Rest)

------------------------------------------------------------------------

## 6. Explainability (Required for TFM Quality)

Use: - **SHAP** for tabular features\
- **LIME** for text explanations\
- Word importance per priority class\
- Feature importance plots

This section strengthens the "Interpretability" requirement.

------------------------------------------------------------------------

## 7. Business Metrics (Critical for Top Grade)

### 7.1 SLA Violations Reduction

Compare predicted vs real priority effect on SLA compliance.

### 7.2 Cost Savings

Assign a penalty cost per SLA breach and compute: \> "The model reduces
SLA violations by X%, saving Y€ per month."

### 7.3 Support Efficiency Metrics

-   Time saved per ticket\
-   Tickets correctly escalated without human intervention\
-   Workload reduction

### 7.4 Scenario Simulation

Examples: - "What if recall of High Priority improves by 10%?"\
- "How many SLA breaches could be prevented yearly?"

------------------------------------------------------------------------

## 8. Impactful Visualizations

Recommended: - Confusion matrix (styled)\
- SHAP summary plot\
- Feature importance ranking\
- Wordcloud by class\
- SLA savings bar chart\
- Sankey diagram for workflow

------------------------------------------------------------------------

## 9. Conceptual Deployment Plan (Required)

### Architecture

-   Preprocessing pipeline\
-   ML model served through API (FastAPI)\
-   Docker container

### Re-training Strategy

-   Monthly re-training\
-   Trigger-based retraining upon drift detection

### Monitoring

-   Model performance metrics\
-   Data drift\
-   SLA drift detection

### MLOps Considerations

-   Versioning\
-   Logging\
-   Alerts

------------------------------------------------------------------------

## 10. Conclusions & Future Improvements

-   Summary of findings\
-   Impact on business\
-   Limitations\
-   Potential extensions:
    -   Automatic ticket type classification\
    -   Entity recognition\
    -   Multi-label classification\
    -   End-to-end workflow automation

------------------------------------------------------------------------

## 📌 Final Notes

This README serves as the structural foundation for the entire TFM
Project 1, ensuring compliance with the requirements while offering
multiple extensions and value-adding components to stand out
academically.
