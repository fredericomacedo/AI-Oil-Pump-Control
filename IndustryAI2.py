# Machine Learning For Production Control
# autor: Frederico Lucio Macedo
# Date: Feb 2019
# Last Updated: July 2023
# This script is part of the initiative of an AI 
# test program to control the production of oil and 
# gas system

import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Function to load data
def load_data():
    with open('data.json', 'r') as file:
        return json.load(file)

# Function to save data
def save_data(data):
    with open('data.json', 'w') as file:
        json.dump(data, file, indent=4)

# Load data
data = load_data()

# Get input for new entry
well_pression = float(input("Enter well_pression: "))
well_temp = float(input("Enter well_temp: "))
well_flow = float(input("Enter well_flow: "))
sumpLevel = float(input("Enter sumpLevel: "))
pump_flow = float(input("Enter pump_flow: "))

# Create the new data point with dummy acceleration value (will be replaced by prediction)
new_data = {
    "well_pression": well_pression,
    "well_temp": well_temp,
    "well_flow": well_flow,
    "sumpLevel": sumpLevel,
    "pump_flow": pump_flow,
    "pump_accelaration": 0
}

# Extract features and target
X = []
y = []

for entry in data:
    features = [
        entry['well_pression'],
        entry['well_temp'],
        entry['well_flow'],
        entry['sumpLevel'],
        entry['pump_flow']
    ]
    target = entry['pump_accelaration']
    X.append(features)
    y.append(target)

X = np.array(X)
y = np.array(y)

# Create and train a linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict acceleration for the new data point
predicted_acceleration = model.predict([[well_pression, well_temp, well_flow, sumpLevel, pump_flow]])
new_data["pump_accelaration"] = predicted_acceleration[0]

# Update the data list
if len(data) == 20:
    data.pop(0)  # Remove the oldest (first) entry
data.append(new_data)

# Save the updated data
save_data(data)
print(f"Predicted Pump Acceleration for new data: {predicted_acceleration[0]}%")
