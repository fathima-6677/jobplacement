import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.naive_bayes import GaussianNB
import evaluate_model as eval_mod
import visualizations as viz

def load_data(filepath):
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    print("\n--- Data Validation ---")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nMissing values:\n", df.isnull().sum())
    print("\nClass distribution ('status'):\n", df['status'].value_counts())
    print("\nDuplicates: ", df.duplicated().sum())
    print("\nData Types:\n", df.dtypes)
    
    return df

def preprocess_data(df):
    # Drop unnecessary columns
    if 'sl_no' in df.columns:
        df = df.drop('sl_no', axis=1)
    if 'salary' in df.columns:
        df = df.drop('salary', axis=1)
        
    # Map target variable
    df['status'] = df['status'].map({'Placed': 1, 'Not Placed': 0})
    
    X = df.drop('status', axis=1)
    y = df['status']
    
    # Identify numerical and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\nNumerical features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return X, y, preprocessor

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'Placement_Data_Full_Class.csv')
    model_path = os.path.join(base_dir, 'models', 'placement_naive_bayes.pkl')
    output_dir = os.path.join(base_dir, 'outputs')
    
    os.makedirs(os.path.join(base_dir, 'models'), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    df = load_data(data_path)
    X, y, preprocessor = preprocess_data(df)
    
    # Generate class distribution plot
    viz.plot_class_distribution(y, output_dir)
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Pipeline creation
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GaussianNB())
    ])
    
    # Hyperparameter tuning using GridSearchCV
    print("\n--- Hyperparameter Tuning ---")
    param_grid = {
        'classifier__var_smoothing': np.logspace(0,-9, num=100)
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1', n_jobs=-1)
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    # 5-fold CV score
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='f1')
    print(f"\n5-fold CV F1-score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Predictions
    y_train_pred = best_model.predict(X_train)
    y_train_prob = best_model.predict_proba(X_train)[:, 1]
    
    y_test_pred = best_model.predict(X_test)
    y_test_prob = best_model.predict_proba(X_test)[:, 1]
    
    # Evaluation
    train_metrics = eval_mod.evaluate_predictions(y_train, y_train_pred, y_train_prob, "Training")
    test_metrics = eval_mod.evaluate_predictions(y_test, y_test_pred, y_test_prob, "Testing")
    
    # Check for overfitting/underfitting
    train_f1 = train_metrics['F1-Score']
    test_f1 = test_metrics['F1-Score']
    
    print("\n--- Model Fit Analysis ---")
    print(f"Training F1: {train_f1:.4f}")
    print(f"Testing F1: {test_f1:.4f}")
    if abs(train_f1 - test_f1) > 0.1:
        if train_f1 > test_f1:
            print("Warning: The model shows signs of overfitting (high training F1, lower testing F1).")
        else:
            print("Warning: The model has unusual performance differences.")
    elif train_f1 < 0.7:
        print("Warning: The model shows signs of underfitting (low F1 on both sets).")
    else:
        print("The model seems to generalize well without significant overfitting or underfitting.")
        
    # Visualizations
    viz.plot_confusion_matrix(y_test, y_test_pred, output_dir)
    viz.plot_roc_curve(y_test, y_test_prob, output_dir)
    
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    train_scores = [train_metrics[m] for m in metrics_names]
    test_scores = [test_metrics[m] for m in metrics_names]
    viz.plot_train_test_comparison(train_scores, test_scores, metrics_names, output_dir)
    
    # Save Model
    joblib.dump(best_model, model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    main()
