# Loan Grade Classification Using Multinomial Logistic Regression

## Project Overview
This project focuses on building a machine learning model to predict loan grades (A to G) based on financial and borrower-related features from the [Lending Club Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club). Accurate loan grading is critical for lenders to assess risk, determine interest rates, and make informed lending decisions. The dataset includes 76 features such as loan amount, interest rate, employment length, and credit history. The goal is to classify loans into their respective grades using a multinomial logistic regression model, addressing class imbalance and ensuring robust performance.
[See code](Final_2.ipynb)
---

## Key Steps & Methodology

### 1. Data Preprocessing
- **Data Loading & Cleaning**:
  - Removed irrelevant columns (e.g., `id`).
  - Scaled numeric features (e.g., `loan_amnt`, `int_rate`) to standardize values.
  - Converted categorical variables (e.g., `grade`, `home_ownership`) into factors for modeling.

- **Feature Engineering**:
  - Created dummy variables for categorical features (`sub_grade`, `hardship_flag`, etc.) using `fastDummies`, removing the original columns to avoid redundancy.

### 2. Handling Class Imbalance
- The target variable `grade` exhibited class imbalance, with fewer samples in lower grades (e.g., F, G). To address this:
  - Applied **SMOTE (Synthetic Minority Oversampling Technique)** to oversample minority classes.
  - Balanced dataset increased from ~709k to ~717k rows, ensuring equitable representation of all grades.

### 3. Model Training
- **Data Splitting**:
  - Split data into 80% training and 20% testing sets using `caret::createDataPartition` to maintain class distribution.
  
- **Algorithm Selection**:
  - Used **multinomial logistic regression** (`nnet::multinom`) due to its suitability for multi-class classification.
  - Trained with **3-fold cross-validation** to optimize generalization.

### 4. Performance Evaluation
- **Training Results**:
  - **Accuracy**: 99.59%
  - **Class-wise Sensitivity**: All classes exceeded 97.5%, with high precision (e.g., 99.8% for Grade A).
  
- **Testing Results**:
  - **Accuracy**: 99.58%
  - **Consistent Performance**: Sensitivity and precision remained stable across classes, demonstrating minimal overfitting.

---

## Technical Highlights
- **Tools & Libraries**:
  - **R** for end-to-end analysis.
  - `caret` for model training and cross-validation.
  - `smotefamily` for addressing class imbalance.
  - `fastDummies` for efficient one-hot encoding.

- **Key Techniques**:
  - Feature scaling, categorical encoding, SMOTE for imbalance correction, and cross-validation.

---

## Results & Insights
- The model excelled at distinguishing between grades, even for rare classes (e.g., Grade G achieved 97.4% sensitivity).
- High balanced accuracy (~98–99% across classes) indicates robustness in real-world scenarios.
- Confusion matrices revealed minimal misclassifications, primarily between adjacent grades (e.g., B vs. C), which is expected due to similar risk profiles.

  <img width="220" alt="Image" src="https://github.com/user-attachments/assets/6ee27bf7-1fbf-45f3-8c50-c8ef1b3b0a17" /> <img width="223" alt="Image" src="https://github.com/user-attachments/assets/a659dccc-6b92-4343-83d0-8192ec1b4d82" />
  

---

## Why This Matters
- **Business Impact**: Accurate loan grading reduces financial risk and enhances decision-making for lenders.
- **Technical Value**: Demonstrates a scalable pipeline for handling imbalanced data and high-dimensional features.

---

## Conclusion
This project showcases a systematic approach to solving a multi-class classification problem, from preprocessing to model deployment. By addressing class imbalance and leveraging cross-validation, the model achieves high accuracy while maintaining interpretability. The workflow is adaptable to other financial risk assessment tasks, highlighting the importance of robust data preparation and algorithm selection.

---

## Skills Demonstrated
- **Data Wrangling**: Cleaning, scaling, and encoding.
- **Imbalanced Data Handling**: SMOTE implementation.
- **Model Development**: Cross-validation, hyperparameter tuning.
- **Tools**: R, `caret`, `smotefamily`, `fastDummies`.

---
