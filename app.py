from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the saved model and preprocessing assets
try:
    assets = joblib.load('heart_disease_model_package.pkl')
    model = assets['model']
    scaler = assets['scaler']
    selected_features = assets['selected_features']
    encoders = assets.get('encoders', {})
    
    # Extract categorical options for the frontend dropdowns
    cat_options = {}
    for col, enc in encoders.items():
        if hasattr(enc, 'classes_'):
            # Convert classes to standard strings to avoid numpy serialization issues
            cat_options[col] = [str(cls) for cls in enc.classes_]
            
except Exception as e:
    print(f"Error loading model: {e}")
    model, scaler, selected_features, encoders, cat_options = None, None, [], {}, {}

@app.route('/')
def home():
    # Pass the selected features and their categorical options to the template
    return render_template('index.html', features=selected_features, cat_options=cat_options)

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return "Model not loaded properly."

    try:
        input_data = {}
        for feature in selected_features:
            val = request.form.get(feature)
            
            # Encode categorical variables if they exist in the encoders dictionary
            if feature in encoders:
                try:
                    val = encoders[feature].transform([val])[0]
                except:
                    val = 0 # Fallback
            else:
                val = float(val) if val else 0.0
                
            input_data[feature] = val

        df_input = pd.DataFrame([input_data])
        
        # Make Prediction
        prediction = model.predict(df_input)[0] 
        probability = model.predict_proba(df_input)[0][1] * 100

        result_text = "High Risk of Heart Disease" if prediction == 1 else "Low Risk of Heart Disease"
        
        return render_template(
            'index.html', 
            prediction_text=f'Result: {result_text}',
            probability_text=f'Probability: {probability:.2f}%',
            features=selected_features,
            cat_options=cat_options
        )

    except Exception as e:
        return f"An error occurred during prediction: {str(e)}"

if __name__ == "__main__":
    # use_reloader=False prevents the freezing issue on Python 3.13
    app.run(debug=True, use_reloader=False)