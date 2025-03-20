import logging
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Writing logs to console and 'app.log'
logging.basicConfig(
    filename="app.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
app.logger.setLevel(logging.DEBUG)
app.logger.propagate = True

@app.before_request
def log_request():
    # POST req with json header
    if request.method == 'POST' and request.is_json:
        data = request.get_json(silent=True)
    else:
        data = None
    app.logger.info(f"Request {request.method} {request.url} | Data: {data}")

@app.route('/')
def index():
    # html page used to render the UI
    return render_template('index.html')

#Model loading
try:
    model = joblib.load("california_rf_model.joblib")
    app.logger.info("Model loaded successfully.")
except Exception as e:
    app.logger.critical(f"Error loading model: {e}", exc_info=True)
    raise RuntimeError("Model loading failed. Check the file path or model integrity.")

# The model expects these features in this exact order:
feature_order = [
    "MedInc", 
    "HouseAge", 
    "AveRooms", 
    "AveBedrms",
    "Population", 
    "AveOccup", 
    "Latitude", 
    "Longitude",
    "PopulationPerHousehold",  
    "RoomsPerPerson"
]

# Pre computed Median values for every feature (in case the user forgets or is not aware of some values to input, these will be taken as the input).
median_values = {
    "MedInc": 3.535, 
    "HouseAge": 29.000, 
    "AveRooms": 5.230, 
    "AveBedrms": 1.105,
    "Population": 1166.000, 
    "AveOccup": 2.818, 
    "Latitude": 34.260, 
    "Longitude": -118.490,
    "PopulationPerHousehold": 3.1, 
    "RoomsPerPerson": 1.1
}

def validate_input(value, feature_name):
    """Checks if the value is numeric or float. in any other case it returns error."""
    if isinstance(value, (int, float)):
        return value
    else:
        raise TypeError(f"{feature_name} must be an integer or a float.")

def format_feature(feature, value, used_median=False):
    """Create a friendly string, optionally noting if a median was used."""
    note = " (median used)" if used_median else ""
    
    if feature == "MedInc":
        # 3.5 ~ $35k
        return f"{value} (≈ {value*10:.2f}k USD){note}"
    elif feature == "Population":
        if value < 1000:
            return f"{int(value)}{note}"
        elif value < 1_000_000:
            return f"{value/1000:.2f}k{note}"
        else:
            return f"{value/1_000_000:.2f}M{note}"
    elif feature == "HouseAge":
        return f"{value} years{note}"
    elif feature == "AveRooms":
        return f"{value} rooms{note}"
    elif feature == "AveBedrms":
        return f"{value} bedrooms{note}"
    elif feature == "AveOccup":
        return f"{value} occupants/HH{note}"
    elif feature == "Latitude":
        return f"{value}° LAT{note}"
    elif feature == "Longitude":
        return f"{value}° LONG{note}"
    elif feature == "RoomsPerPerson":
        return f"{value} RPP{note}"
    elif feature == "PopulationPerHousehold":
        return f"{value} (set = AveOccup){note}"
    else:
        return f"{value}{note}"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        app.logger.info(f"Received request in /predict: {request.get_json(silent=True)}")
        data = request.get_json(force=True) 

        used_values_raw = {}
        used_values_formatted = []
        features_list = []

        # Build features in the order required by our model
        for feature in feature_order:
            if feature == "PopulationPerHousehold": #this feature I added is same as the AvgOccup feature, hence I combined them both.
                # We set this to whatever AveOccup is, after it’s resolved
                validated_value = used_values_raw["AveOccup"]
                used_median = False
            else:
                # If the key is present but the user left it blank, we'll get `None`.
                raw_val = data.get(feature, None)

                # If user gave us None or didn't provide it, we use the median
                if raw_val is None:
                    raw_val = median_values[feature]
                    used_median = True
                else:
                    used_median = False

                # validating the input
                validated_value = validate_input(raw_val, feature)

            used_values_raw[feature] = validated_value

            # A string showing the feature + final value used
            display_str = f"{feature}: {format_feature(feature, validated_value, used_median)}"
            used_values_formatted.append(display_str)

            features_list.append(validated_value)

        # Convert features to a DataFrame for scikit-learn
        df = pd.DataFrame([features_list], columns=feature_order)

        # Our models prediction
        prediction = model.predict(df)
        app.logger.info(f"Prediction: {prediction[0]} for input {features_list}")

        house_value = float(prediction[0])*100000


        return jsonify({
            "prediction": house_value,
            "used_values": used_values_raw,
            "used_values_readable": used_values_formatted
        })

    except Exception as e:
        app.logger.critical(f"Unexpected error in prediction: {e}", exc_info=True)
        return jsonify({"error": "Internal server error. Please try again later."}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.critical(f"Unhandled Exception: {e}", exc_info=True)
    return jsonify({"error": "Something went wrong. Please contact support."}), 500

if __name__ == '__main__':
    app.run(debug=True)
