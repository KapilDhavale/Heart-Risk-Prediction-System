from flask import Flask, render_template, request
import joblib  # Import joblib to load the model and scaler

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('best_rf_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    features = [
        float(request.form['age']),
        float(request.form['sex']),
        float(request.form['cp']),
        float(request.form['trestbps']),
        float(request.form['chol']),
        float(request.form['fbs']),
        float(request.form['restecg']),
        float(request.form['thalach']),
        float(request.form['exang']),
        float(request.form['oldpeak']),
        float(request.form['slope']),
        float(request.form['ca']),
        float(request.form['thal'])
    ]

    # Scale the features
    scaled_features = scaler.transform([features])

    # Make prediction
    prediction = model.predict(scaled_features)[0]

    return render_template('result.html', prediction=prediction, **dict(zip([
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
        'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ], features)))

if __name__ == '__main__':
    app.run(debug=True)
