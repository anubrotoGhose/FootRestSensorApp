from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import random
import base64
import io
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load trained model and scaler
model = load_model('final_acupoint_model.keras')
scaler = joblib.load('new_standard_scaler.pkl')

# Define foot types and acupoints
foot_types = ["Flat", "Normal", "High Arch"]
acupoints = ["Kidney", "Liver", "Stomach"]

# Function to generate synthetic pressure map
def generate_pressure_map(foot_length, foot_width, foot_type, acupoint):
    grid_size = 10
    pressure_map = np.zeros((grid_size, grid_size))
    foot_center_x = random.randint(3, 6)
    foot_center_y = random.randint(3, 6)

    pressure_variance = {"Flat": 1.2, "Normal": 1.0, "High Arch": 0.8}
    variance = pressure_variance.get(foot_type, 1.0)

    for i in range(grid_size):
        for j in range(grid_size):
            distance = np.sqrt((i - foot_center_x) ** 2 + (j - foot_center_y) ** 2)
            pressure_map[i, j] += np.exp(-distance**2 / (2 * (foot_width / 10) ** 2 * variance))

    pressure_map /= pressure_map.max()
    pressure_map += np.random.normal(0, 0.05, (grid_size, grid_size))
    pressure_map = np.clip(pressure_map, 0, 1)

    acupoint_locations = {"Kidney": (-1, 0), "Liver": (1, 1), "Stomach": (0, -1)}
    dx, dy = acupoint_locations.get(acupoint, (0, 0))
    acupoint_x, acupoint_y = np.clip([foot_center_x + dx, foot_center_y + dy], 0, grid_size - 1)

    return pressure_map, (foot_center_x, foot_center_y), (acupoint_x, acupoint_y)

# Function to preprocess input
def preprocess_input(pressure_map, foot_type, acupoint):
    flattened_map = pressure_map.flatten().tolist()
    input_data = pd.DataFrame({"Foot Type": [foot_type], "Acupoint": [acupoint]})
    for i in range(len(flattened_map)):
        input_data[f'Pressure_{i+1}'] = flattened_map[i]

    input_data = pd.get_dummies(input_data, columns=['Foot Type', 'Acupoint'], drop_first=True)
    missing_cols = set(scaler.feature_names_in_) - set(input_data.columns)
    for col in missing_cols:
        input_data[col] = 0

    input_data = input_data[scaler.feature_names_in_]
    input_data = scaler.transform(input_data)
    return input_data

# Function to generate and encode visualization
def generate_visualization(pressure_map, foot_center, predicted_acupoint):
    plt.imshow(pressure_map, cmap='hot', interpolation='nearest')
    plt.colorbar(label="Pressure Intensity")

    # Plot Foot Center and Predicted Acupoint outside the if condition
    plt.scatter(*foot_center[::-1], c='green', s=100, marker='x', label='Foot Center')
    plt.scatter(*predicted_acupoint[::-1], c='red', s=100, marker='o', label='Predicted Acupoint')

    # Plot the Foot Border if border points exist
    border_mask = pressure_map > 0
    border_x, border_y = np.where(border_mask)

    if border_x.size > 0 and border_y.size > 0:
        plt.scatter(border_y, border_x, c='blue', s=30, label='Foot Border')

    plt.legend(loc='best')
    plt.grid(True)

    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png', bbox_inches='tight')
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.read()).decode()
    plt.close()
    
    return img_base64

# Route for form input
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        foot_type = request.form["foot_type"]
        acupoint = request.form["acupoint"]
        foot_length = random.choice(range(22, 30))
        foot_width = random.choice(range(8, 14))

        pressure_map, foot_center, _ = generate_pressure_map(foot_length, foot_width, foot_type, acupoint)
        input_data = preprocess_input(pressure_map, foot_type, acupoint)
        predicted_coords = model.predict(input_data)
        predicted_acupoint = (round(predicted_coords[0][0]), round(predicted_coords[0][1]))

        img_base64 = generate_visualization(pressure_map, foot_center, predicted_acupoint)

        return render_template("result.html", foot_type=foot_type, acupoint=acupoint, 
                               foot_center=foot_center, predicted_acupoint=predicted_acupoint, 
                               visualization=img_base64)
    
    return render_template("index.html", foot_types=foot_types, acupoints=acupoints)