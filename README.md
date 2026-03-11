# WalkSafe - Pedestrian Safety Prediction System

WalkSafe is a machine learning based application that predicts the safety level of a location for pedestrians.

The model analyzes environmental factors and predicts whether the area is:

Safe
Moderate
Unsafe

## Features Used
Lighting conditions
Crowd density
Incident count
Time of day

## Algorithm
Random Forest Classifier

## Technologies
Python
Flask
Scikit-learn
Pandas
HTML/CSS

## Project Architecture

Dataset → Preprocessing → Model Training → Flask Web App → Safety Prediction

## Installation

Install dependencies

pip install -r requirements.txt

Train the model

python train_model.py

Run the application

python app.py

Open browser

http://127.0.0.1:5000