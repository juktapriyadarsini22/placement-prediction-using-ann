from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model
model = load_model('./models/placement_model.h5')

# Load the LabelEncoders and StandardScaler
with open('./models/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('./models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the dataset and process it consistently
data = pd.read_csv('./data/data.csv')

# Encode categorical variables and apply one-hot encoding
data_encoded = pd.get_dummies(data, columns=['gender', 'class', 'course', 'test preparation course'])

# Separate features and target variable
X = data_encoded.drop('placed', axis=1)
y = data_encoded['placed'].map({'yes': 1, 'no': 0})

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the LIME explainer
explainer = LimeTabularExplainer(
    training_data=X_scaled,
    feature_names=X.columns,
    class_names=['not placed', 'placed'],
    mode='classification'
)

@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract input data from the form
        gender = request.form['gender']
        student_class = request.form['class']
        course = request.form['course']
        test_prep = request.form['test_prep']
        aptitude_score = float(request.form['aptitude_score'])
        coding_score = float(request.form['coding_score'])
        gd_score = float(request.form['gd_score'])

        # Create DataFrame with the input data
        input_data = pd.DataFrame({
            'gender': [gender],
            'class': [student_class],
            'course': [course],
            'test preparation course': [test_prep],
            'aptitude score': [aptitude_score],
            'coding score': [coding_score],
            'group discusion score': [gd_score]
        })

        # Encode categorical variables for prediction data
        input_data_encoded = pd.get_dummies(input_data, columns=['gender', 'class', 'course', 'test preparation course'])
        
        # Ensure the prediction data has the same columns as the training data
        input_data_encoded = input_data_encoded.reindex(columns=X.columns, fill_value=0)
        
        # Normalize the input data
        input_data_scaled = scaler.transform(input_data_encoded)

        # Make prediction
        prediction = (model.predict(input_data_scaled) > 0.5).astype(int)
        result = 'yes' if prediction[0][0] == 1 else 'no'

        # Generate LIME explanation if not placed
        if result == 'no':
            # Get explanation for the input data
            explanation = explainer.explain_instance(input_data_scaled[0], model.predict_proba)
            explanation_html = explanation.as_html()
        else:
            explanation_html = None

        return render_template('index2.html', prediction_text=f'Will be placed: {result}', explanation_html=explanation_html)

if __name__ == "__main__":
    app.run(debug=True)
