# Job Placement Prediction using Naive Bayes

## Problem Statement
Predicting student placement is crucial for educational institutions to provide targeted training and improve student employability.

## Objective
The objective of this project is to build a machine learning model that accurately predicts whether a student will be placed or not based on their academic background and other attributes.

## Dataset Description
The dataset contains information about students' academic percentages from 10th grade to MBA, along with their degree types, boards of education, gender, work experience, and employability test scores.

## Features Used
- `gender`: Gender of the candidate
- `ssc_p`: Secondary Education percentage (10th)
- `ssc_b`: Board of Education for 10th
- `hsc_p`: Higher Secondary Education percentage (12th)
- `hsc_b`: Board of Education for 12th
- `hsc_s`: Specialization in Higher Secondary Education
- `degree_p`: Degree percentage
- `degree_t`: Degree type
- `workex`: Work experience
- `etest_p`: Employability test percentage
- `specialisation`: Post Graduation (MBA) specialization
- `mba_p`: MBA percentage

## Features Removed
- `sl_no`: Serial number, which is just an identifier and provides no predictive value.
- `salary`: Future salary, which is only available *after* placement. Including it would cause data leakage.

## Naive Bayes Explanation
Gaussian Naive Bayes is a probabilistic classifier based on applying Bayes' theorem with strong (naive) independence assumptions between the features. It assumes the continuous features follow a Gaussian (normal) distribution.

## Preprocessing
- **Numerical Features**: Missing values are imputed with the median, followed by standardization using `StandardScaler`.
- **Categorical Features**: Missing values are imputed with the most frequent value, followed by one-hot encoding using `OneHotEncoder`.
All preprocessing steps are encapsulated within a `ColumnTransformer` inside a scikit-learn `Pipeline` to prevent data leakage during cross-validation.

## Train/Test Split
The data is split into 80% training data and 20% testing data, stratified on the target variable with a fixed random state for reproducibility.

## Cross-validation & Hyperparameter Tuning
We use 5-fold Stratified K-Fold cross-validation. We optimize the `var_smoothing` parameter of the GaussianNB classifier using `GridSearchCV` over a logarithmic space, prioritizing the F1 score.

## Evaluation Metrics
- **Accuracy**: Overall correctness of the model.
- **Precision**: Accuracy of positive predictions.
- **Recall**: Ability to find all positive instances.
- **F1-Score**: Harmonic mean of precision and recall.
- **ROC-AUC**: Area under the receiver operating characteristic curve.

## Overfitting/Underfitting Discussion
The model achieved a training F1 score of **0.9032** and a testing F1 score of **0.9062**. 
Since the testing F1 score is very close to the training F1 score (and both are high), the model demonstrates strong generalization without any notable signs of overfitting or underfitting.

## Setup Instructions

### How to Install Dependencies
```bash
pip install -r requirements.txt
```

### How to Train the Model
```bash
python src/train_model.py
```
*This will evaluate the model, save visualizations in `outputs/`, and save the trained pipeline in `models/`.*

### How to Run Predictions
```bash
python src/predict.py
```

### How to Launch the Streamlit Application
```bash
streamlit run app/app.py
```

## Expected Project Output
- A trained model saved as `models/placement_naive_bayes.pkl`.
- Four plots saved in `outputs/` (Confusion matrix, ROC curve, Train/Test comparison, Class distribution).
- A web interface to enter student data and receive placement predictions.

## Limitations
- The dataset size is relatively small (215 records), which may limit the model's ability to learn highly complex patterns.
- Naive Bayes assumes feature independence, which may not hold perfectly in reality (e.g., 10th and 12th percentages might be correlated).

## Future Improvements
- Collect more data to increase robustness.
- Feature engineering (e.g., creating interaction terms between percentages).
- Try other algorithms like Random Forest or Gradient Boosting for comparison.
