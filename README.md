# California Housing Price Prediction

This project builds a machine learning model to predict housing prices in California based on various housing and demographic features. The trained model is deployed via a Flask API.

## Features
- Data preprocessing and feature engineering
- Model training using Random Forest Regressor
- Hyperparameter tuning using GridSearchCV
- Flask API for real-time predictions

## Installation

### Prerequisites
Ensure you have Python 3 installed.

### Install Dependencies
Run the following command to install necessary libraries:
```bash
pip install -r requirements.txt
```

## Running the Model Training
Execute the Jupyter Notebook to preprocess data, train the model, and save it.
```bash
jupyter notebook california_housing.ipynb
```

## Running the Flask API
Start the Flask application to serve the trained model.
```bash
python myapp.py
```

## Making Predictions
Once the API is running, send a POST request to `http://127.0.0.1:5000/predict` with the required feature values.

### Example Request
```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{
         "MedInc": 3.5, 
         "HouseAge": 30, 
         "AveRooms": 5.0, 
         "AveBedrms": 1.1, 
         "Population": 1000, 
         "AveOccup": 2.5, 
         "Latitude": 34.0, 
         "Longitude": -118.0, 
         "PopulationPerHousehold": 3.0, 
         "RoomsPerPerson": 1.2
     }'
```

### Example Response
```json
{"prediction": 2.45}
```

## Project Structure
```
├── california_housing.ipynb   # Jupyter Notebook for training and evaluation
├── myapp.py                   # Flask API for model inference
├── california_rf_model.joblib  # Saved trained model
├── requirements.txt            # Dependencies
├── README.md                   # Project instructions
```
