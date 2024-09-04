from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('./models/placement_model.h5')

# Load the LabelEncoders and StandardScaler
with open('./models/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('./models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        gender = request.form['gender']
        student_class = request.form['class']
        course = request.form['course']
        test_prep = request.form['test_prep']
        aptitude_score = float(request.form['aptitude_score'])
        coding_score = float(request.form['coding_score'])
        gd_score = float(request.form['gd_score'])

        # Create a DataFrame with the input data
        input_data = pd.DataFrame({
            'gender': [gender],
            'class': [student_class],
            'course': [course],
            'test preparation course': [test_prep],
            'aptitude score': [aptitude_score],
            'coding score': [coding_score],
            'group discusion score': [gd_score]
        })

        # Encode and normalize the input data
        for column in ['gender', 'class', 'course', 'test preparation course']:
            input_data[column] = label_encoders[column].transform(input_data[column])
        input_data_scaled = scaler.transform(input_data)

        # Make the prediction
        prediction = (model.predict(input_data_scaled) > 0.5).astype(int)
        result = 'yes' if prediction[0][0] == 1 else 'no'

        return render_template('index.html', prediction_text=f'Will be placed: {result}')

if __name__ == "__main__":
    app.run(debug=True)
