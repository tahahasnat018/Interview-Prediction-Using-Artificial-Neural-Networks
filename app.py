from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load the Keras model (ensure the model.h5 file is in the same directory)
try:
    model = load_model('model.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Single route for the form and predictions
@app.route('/', methods=['GET', 'POST'])
def interview_form():
    if request.method == 'POST':
        try:
            # Extract form data from request
            form_data = request.form

            # Parse input values from the form
            features = [
                int(form_data['age']),
                int(form_data['candidate_status']),
                int(form_data['gender']),
                int(form_data['fluency']),
                int(form_data['mother_tongue_influence']),
                int(form_data['acquaintance']),
                int(form_data.get('currently_employed', 0)),   # Optional field
                int(form_data.get('willing_to_relocate', 0)),   # Optional field
                int(form_data.get('confidence_intro', 0)),
                int(form_data.get('confidence_topic', 0)),
                int(form_data.get('confidence_ppt', 0)),
                int(form_data.get('confidence_sales', 0)),
                int(form_data.get('structured_thinking_region', 0)),
                int(form_data.get('structured_thinking_ppt', 0)),
                int(form_data.get('structured_thinking_pitch', 0)),
                int(form_data.get('regional_fluency_topic', 0)),
                int(form_data.get('regional_fluency_ppt', 0)),
                int(form_data.get('regional_fluency_sales', 0))
            ]

            # Prepare input data for the model
            input_data = np.array([features])

            # Debug: Print input data
            print("Input Data:", input_data)

            # Ensure the model is loaded
            if model is None:
                raise ValueError("Model is not loaded. Please check the model file.")

            # Make prediction
            prediction = model.predict(input_data)

            # Debug: Print prediction output
            print("Prediction Output:", prediction)

            # Get the prediction class based on a threshold (assuming output is a single probability)
            prediction_class = 1 if prediction[0][0] >= 0.5 else 0
            interview_verdict = 'Selected' if prediction_class == 1 else 'Rejected'

            return render_template('index.html', prediction=interview_verdict)

        except Exception as e:
            # Log the error
            print(f"Error during prediction: {e}")
            return render_template('index.html', error=f"Error: {str(e)}")

    # Render the form for GET requests
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
