import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle

# Load dataset 
data = pd.read_csv('student.csv')
data.head()

features = ['studytime', 'failures', 'absences', 'G1', 'G2']
data = data.dropna(subset=features + ['G3'])  

X = data[features].astype(float)
y = data['G3'].apply(lambda x: 1 if x >= 10 else 0)  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_s, y_train)

preds = model.predict(X_test_s)
acc = accuracy_score(y_test, preds)
print(f"Test Accuracy: {acc:.3f}")

with open('student_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

import json
with open('features.json', 'w') as f:
    json.dump(features, f)
print("Saved student_model.pkl, scaler.pkl, features.json")
