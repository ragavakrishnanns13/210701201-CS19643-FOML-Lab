from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the pre-trained machine learning model
model = tf.keras.models.load_model(r"/Users/ragavakrishnanns/Downloads/210701201-FOML/Mini-Project/Code/tumour_model.h5")
print("hello",model)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    name = request.form['name']
    age = request.form['age']
    csv_file = request.files['csv']

    # Read CSV file
    # data = pd.read_csv(csv_file,encoding='utf-8',errors='ignore')
    try:
        data = pd.read_csv(csv_file, encoding='utf-8')
    except UnicodeDecodeError as e:
        # Handle the error gracefully
        print("Error reading CSV file:", e)
        # Render an error message to the user
        return render_template('error.html', error_message="Error reading CSV file: " + str(e))
    

    # Preprocess data (if needed)
    # For example, you might need to drop unnecessary columns, handle missing values, or scale the data
    # You should preprocess the data in the same way as you preprocessed the data for training the model

    # Make predictions
    predictions = model.predict(data)

    # Assuming the model predicts the probability of tumor
    # tumor_probability = np.mean(predictions)
    tumor_probability = round(predictions[0][0]*100,2)

    return render_template('result.html', name=name, age=age, tumor_probability=tumor_probability)

if __name__ == '__main__':
    app.run(debug=True)