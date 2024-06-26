# app.py
from myPipeline import *
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify
import joblib
import pandas as pd


# Initialize the Flask app
app = Flask(__name__)

# Load the model
model = joblib.load('model.joblib')

# Preprocess the input data
def preprocess_input(data):
    # Transform the input data into a suitable format
    # Here we assume the data is already preprocessed as per the training procedure
    df = pd.DataFrame(data)
    return df

# Define the root route
@app.route('/')
def index():
    
    return "Welcome to the Diamond Price Prediction API!"


# First use case: price prediction function
@app.route('/predict', methods=['POST'])
def predict():
    
    # Get JSON data from the request
    data = request.get_json(force=True)
    
    # Preprocess the input data
    input_data = preprocess_input(data)
    
    # Make predictions
    predictions = model.predict(input_data)
    
    # Return predictions as JSON
    return jsonify({'predictions': predictions.tolist()})
    
#Second use case: given the features of a diamond, 
#return n samples from the training dataset with the same cut, 
#color, and clarity, and the most similar weight.

@app.route('/similarity',methods=['POST'])

def find_similar(training_df,cut, color, clarity, weight,n):
    
    mask = [(training_df['cut'] == cut) & 
        (training_df['color'] == color) & 
        (training_df['clarity'] == clarity)]
    
    df_filtered = training_df[mask]

    # Sort by the absolute difference value in weight
    df_filtered['weight_diff'] = (df_filtered['carat'] - weight).abs()
    similar_data = df_filtered.sort_values(by='weight_diff',ascending=True)
    
    # Return the first n samples
    return similar_data.head(n)


def similarity():

    # Get JSON data from the request
    data = request.get_json(force=True)
    
    # Preprocess the input data
    input_data = preprocess_input(data)
    
    #train - test splitting
    input_data_trained,input_data_test = train_test_split(input_data,test_size=0.2,random_state=42)
    
    #similar samples, given a feature such as diamond 
    similar_samples = find_similar(input_data_trained,'Ideal','D','IF',0.5)
    
    return jsonify({f'Similar data: {similar_samples.tolist()}'})


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
