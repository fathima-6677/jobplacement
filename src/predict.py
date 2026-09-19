import os
import joblib
import pandas as pd

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
    return joblib.load(model_path)

def predict_placement(model, student_data):
    """
    Predicts the placement status for a given student.
    student_data should be a dictionary containing the necessary features.
    """
    df = pd.DataFrame([student_data])
    
    # Preprocessing is handled by the pipeline!
    prediction = model.predict(df)[0]
    probabilities = model.predict_proba(df)[0]
    
    status = "Placed" if prediction == 1 else "Not Placed"
    prob_placed = probabilities[1] * 100
    prob_not_placed = probabilities[0] * 100
    
    return status, prob_placed, prob_not_placed

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'models', 'placement_naive_bayes.pkl')
    
    try:
        model = load_model(model_path)
        
        # Sample student
        sample_student = {
            'gender': 'M',
            'ssc_p': 85.0,
            'ssc_b': 'Central',
            'hsc_p': 75.0,
            'hsc_b': 'Central',
            'hsc_s': 'Science',
            'degree_p': 80.0,
            'degree_t': 'Sci&Tech',
            'workex': 'Yes',
            'etest_p': 85.0,
            'specialisation': 'Mkt&Fin',
            'mba_p': 75.0
        }
        
        print("Predicting for sample student:")
        for k, v in sample_student.items():
            print(f"  {k}: {v}")
            
        status, p_placed, p_not_placed = predict_placement(model, sample_student)
        
        print(f"\nPrediction: {status}")
        print(f"Probability of being Placed: {p_placed:.2f}%")
        print(f"Probability of Not being Placed: {p_not_placed:.2f}%")
        
    except Exception as e:
        print(f"Error: {e}")
