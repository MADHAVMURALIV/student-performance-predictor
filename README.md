# Student Performance Predictor

A machine learning web application that predicts whether a student is likely to pass or fail based on academic and behavioral features.

## Overview
This project uses a trained classification model to analyze student data such as study time, past failures, absences, and previous grades. The application provides individual predictions, batch predictions from CSV files, and visual insights through an interactive dashboard.

## Features
- Individual student performance prediction
- Pass/Fail probability with confidence score
- Batch prediction using CSV upload
- Interactive insights dashboard with visualizations
- Feature importance visualization (if supported by model)

## Tech Stack
- Python
- Streamlit
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn

## Project Structure
- `app_1.py` – Streamlit web application
- `model_1.py` – Model training and artifact generation
- `student_model.pkl` – Trained ML model
- `scaler.pkl` – Feature scaling object
- `features.json` – Input feature configuration
- `student.csv` – Dataset
- `requirements.txt` – Dependencies

## Setup Instructions
```bash
pip install -r requirements.txt
<img width="1919" height="1079" alt="Screenshot 2025-09-19 192027" src="https://github.com/user-attachments/assets/bb1a6042-8226-4c91-bfac-6b2cbbf81b86" />
<img width="1919" height="1079" alt="Screenshot 2025-09-19 192000" src="https://github.com/user-attachments/assets/e7b5ffe8-36da-47ba-accc-1a144de0b13e" />

