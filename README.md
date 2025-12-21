# Student Performance Predictor

A machine learning–based web application that predicts whether a student is likely to **pass or fail** based on academic and behavioral factors, with interactive visual insights.

## Overview
This project analyzes student data such as study time, past failures, absences, and previous grades to predict academic performance.  
The application supports **single predictions**, **batch predictions via CSV**, and an **insights dashboard** for exploratory analysis.

## Features
- Individual student performance prediction (Pass / Fail)
- Confidence score for each prediction
- Batch prediction using CSV upload
- Downloadable prediction results
- Interactive insights dashboard
- Correlation heatmap and data visualizations
- Feature importance visualization

## Tech Stack
- **Python**
- **Streamlit** (web interface)
- **Scikit-learn** (ML model)
- **Pandas, NumPy**
- **Matplotlib, Seaborn**

## Project Structure
- `app_1.py` – Streamlit web application
- `model_1.py` – Model training and artifact generation
- `student_model.pkl` – Trained classification model
- `scaler.pkl` – Feature scaler
- `features.json` – Input feature configuration
- `student.csv` – Dataset
- `requirements.txt` – Project dependencies

## Input Features
- Study time per week  
- Number of past failures  
- Absences  
- Grade 1 (G1)  
- Grade 2 (G2)  

## Setup Instructions
```bash
pip install -r requirements.txt
<img width="1917" height="936" alt="Screenshot 2025-11-01 173311" src="https://github.com/user-attachments/assets/afc39451-cf56-4c2b-adb4-0470d1817e5f" />
<img width="1909" height="940" alt="Screenshot 2025-11-01 173257" src="https://github.com/user-attachments/assets/bc918520-ea03-40b2-9f8a-4a8a02ca5510" />
<img width="1918" height="940" alt="Screenshot 2025-11-01 173054" src="https://github.com/user-attachments/assets/0d3c4f48-6b93-476a-a10a-3d1523caaec3" />
<img width="1916" height="957" alt="Screenshot 2025-11-01 173338" src="https://github.com/user-attachments/assets/cdb2f76b-35e1-46a6-ab00-f1e4ac967ebd" />
<img width="1919" height="938" alt="Screenshot 2025-11-01 173225" src="https://github.com/user-attachments/assets/9dac4672-a8ea-4a7f-bf8a-bf51eba130ef" />
<img width="1919" height="946" alt="Screenshot 2025-11-01 173143" src="https://github.com/user-attachments/assets/20c1f87d-e6a3-43b9-bc82-ba6798810226" />
<img width="1918" height="953" alt="Screenshot 2025-11-01 173032" src="https://github.com/user-attachments/assets/48a7167a-f41d-485a-b2e5-1b28259293ab" />
<img width="1919" height="1079" alt="Screenshot 2025-11-01 172912" src="https://github.com/user-attachments/assets/e3f9d79a-ef48-4a6d-8695-9c5cbbb3205f" />
