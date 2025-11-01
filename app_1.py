import streamlit as st
import pickle
import numpy as np
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -------------- CONFIG -----------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
)

# -------------- LOAD MODEL -----------------
@st.cache_data
def load_artifacts():
    with open('student_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('features.json', 'r') as f:
        features = json.load(f)
    return model, scaler, features

model, scaler, features = load_artifacts()

# -------------- GLOBAL STYLING -----------------
st.markdown("""
<style>
/* 🌈 Background gradient */
.stApp {
    background: linear-gradient(to bottom right, #d3f0f7, #f8d7e3);
    color: #1e293b;
    font-family: 'Poppins', sans-serif;
}

/* --- Fix file uploader box (fully visible) --- */
div[data-testid="stFileUploader"] > section[data-testid="stFileUploaderDropzone"] {
    background-color: rgba(255, 255, 255, 0.95) !important;
    color: #000 !important;
    border: 2px dashed #4CAF50 !important;
    border-radius: 15px !important;
    padding: 1.2rem !important;
    transition: all 0.3s ease-in-out !important;
}

/* Hover glow for upload area */
div[data-testid="stFileUploader"] > section[data-testid="stFileUploaderDropzone"]:hover {
    border: 2px dashed #00bcd4 !important;
    box-shadow: 0 0 15px rgba(0, 188, 212, 0.4) !important;
    background-color: #fff !important;
}

/* Upload text */
div[data-testid="stFileUploaderDropzone"] p {
    color: #000 !important;
    font-weight: 500 !important;
    text-align: center !important;
}

/* Browse button */
div[data-testid="stFileUploader"] button {
    background-color: #4CAF50 !important;
    color: #fff !important;
    border-radius: 8px !important;
    padding: 0.5rem 1rem !important;
    font-weight: 600 !important;
    border: none !important;
    transition: all 0.2s ease-in-out !important;
}
div[data-testid="stFileUploader"] button:hover {
    background-color: #45a049 !important;
    transform: scale(1.05) !important;
}

/* 🧭 Sidebar */
[data-testid="stSidebar"] {
    background-color: #111827;
}
[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}

/* 🏷️ General text */
h1, h2, h3, label, p, span, div {
    color: #1e293b !important;
}

/* ✨ Main card container */
.section {
    background-color: #ffffff;
    padding: 25px 35px;
    margin: 20px 0px;
    border-radius: 20px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
}

/* 🎨 Buttons */
.stButton > button {
    background: linear-gradient(90deg, #0ea5e9, #22c55e);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 24px;
    font-size: 16px;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #0284c7, #16a34a);
    transform: scale(1.03);
}

/* ✅ Success and ❌ Error Boxes */
.success-box {
    background-color: #dcfce7;
    color: #14532d;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    font-size: 18px;
    margin-top: 15px;
}
.error-box {
    background-color: #fee2e2;
    color: #7f1d1d;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    font-size: 18px;
    margin-top: 15px;
}
</style>
""", unsafe_allow_html=True)

# -------------- SIDEBAR NAVIGATION -----------------
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", ["🎯 Predict", "📂 Batch Upload", "📈 Insights", "ℹ️ About"])

# -------------- HERO HEADER -----------------
st.markdown("""
<div class='main'>
    <h1 style='text-align:center;'>🎓 Student Performance Predictor</h1>
    <p style='text-align:center; color:#555; font-size:1.1rem;'>
    A smart machine learning app that predicts student outcomes based on study habits and academic performance.
    </p>
    <hr>
""", unsafe_allow_html=True)

# -------------- PAGE 1: PREDICT -----------------
if page == "🎯 Predict":
    st.subheader("🧮 Enter Student Details")
    vals = {}
    col1, col2 = st.columns(2)
    for i, feat in enumerate(features):
        with (col1 if i % 2 == 0 else col2):
            if feat in ['G1', 'G2']:
                vals[feat] = st.slider(f"{feat} (0-20)", 0, 20, 10)
            elif feat == 'studytime':
                vals[feat] = st.slider("Study Time (1= <2hrs, 4= >10hrs)", 1, 4, 2)
            elif feat == 'failures':
                vals[feat] = st.number_input("Past Failures", 0, 5, 0)
            elif feat == 'absences':
                vals[feat] = st.number_input("Absences", 0, 100, 5)
            else:
                vals[feat] = st.number_input(feat, value=0.0)

    st.markdown("### 🎯 Predict Result")

    if st.button("🔍 Predict Performance"):
        x = np.array([[vals[f] for f in features]], dtype=float)
        x_s = scaler.transform(x)
        prob = model.predict_proba(x_s)[0][1]
        pred = 1 if prob >= 0.5 else 0

        if pred == 1:
            st.markdown(
                f"""
                <div style='background:#E8F8F5; padding:25px; border-radius:15px; text-align:center;'>
                    <h2 style='color:#148F77;'>✅ PASS</h2>
                    <h4>Confidence: {prob*100:.1f}%</h4>
                    <p>🎉 Great work! The student is performing consistently.</p>
                </div>
                """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(
                f"""
                <div style='background:#FDEDEC; padding:25px; border-radius:15px; text-align:center;'>
                    <h2 style='color:#C0392B;'>❌ FAIL</h2>
                    <h4>Confidence: {(1-prob)*100:.1f}%</h4>
                    <p>⚠️ The student might need academic support and attention.</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("#### 📊 Confidence Level")
        st.progress(int(prob * 100))

        if hasattr(model, 'feature_importances_'):
            st.markdown("#### 🔎 Feature Importance")
            fi = model.feature_importances_
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.barh(features, fi, color="#3498DB")
            ax.set_xlabel("Importance")
            ax.set_ylabel("Features")
            plt.tight_layout()
            st.pyplot(fig)

# -------------- PAGE 2: BATCH UPLOAD -----------------
elif page == "📂 Batch Upload":
    st.subheader("📁 Batch Prediction from CSV")
    st.write(f"Upload a CSV with columns: **{', '.join(features)}**")
    csv_file = st.file_uploader("Upload CSV", type=['csv'])
    if csv_file:
        df = pd.read_csv(csv_file)
        if not set(features).issubset(df.columns):
            st.error("Uploaded CSV missing required columns.")
        else:
            X = df[features].astype(float)
            Xs = scaler.transform(X)
            probs = model.predict_proba(Xs)[:, 1]
            preds = (probs >= 0.5).astype(int)
            df['pass_probability'] = probs
            df['predicted_pass'] = preds
            st.success("✅ Batch predictions generated successfully!")
            st.dataframe(df.head())
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")

# -------------- PAGE 3: INSIGHTS -----------------
elif page == "📈 Insights":
    st.markdown("<h2 style='text-align:center; color:#4CAF50;'>📊 Student Performance Insights Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#f0f0f0;'>Upload a dataset to explore how different factors influence student grades.</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📁 Upload Student Dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Dataset uploaded successfully!")

        st.markdown("<h4 style='color:#FFD700;'>📋 Dataset Preview</h4>", unsafe_allow_html=True)
        st.dataframe(df.head(), use_container_width=True)

        # ----- FIXED SECTION -----
        st.markdown("<h4 style='color:#FFD700;'>🎯 Correlation Heatmap</h4>", unsafe_allow_html=True)
        numeric_df = df.select_dtypes(include=['int64', 'float64'])

        if numeric_df.shape[1] > 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("⚠️ No numeric columns found for correlation heatmap.")
        # --------------------------

        st.markdown("<h4 style='color:#FFD700;'>📘 Study Time vs Average Final Grade (G3)</h4>", unsafe_allow_html=True)
        if 'studytime' in df.columns and 'G3' in df.columns:
            avg_score = df.groupby('studytime')['G3'].mean().reset_index()
            fig, ax = plt.subplots()
            sns.barplot(data=avg_score, x='studytime', y='G3', palette="mako", ax=ax)
            ax.set_xlabel("Study Time", color="white")
            ax.set_ylabel("Average G3 Score", color="white")
            ax.tick_params(colors="white")
            st.pyplot(fig)
        else:
            st.warning("⚠️ Columns 'studytime' and 'G3' not found in dataset.")

        st.markdown("<h4 style='color:#FFD700;'>📉 Absences vs Final Grade (G3)</h4>", unsafe_allow_html=True)
        if 'absences' in df.columns and 'G3' in df.columns:
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x='absences', y='G3', color="#66c2a5", ax=ax)
            ax.set_xlabel("Absences", color="white")
            ax.set_ylabel("Final Grade (G3)", color="white")
            ax.tick_params(colors="white")
            st.pyplot(fig)
        else:
            st.warning("⚠️ Columns 'absences' and 'G3' not found in dataset.")

        st.markdown("<h4 style='color:#FFD700;'>💡 Quick Insights</h4>", unsafe_allow_html=True)
        if 'studytime' in df.columns and 'G3' in df.columns:
            high_study = df[df['studytime'] > df['studytime'].median()]['G3'].mean()
            low_study = df[df['studytime'] <= df['studytime'].median()]['G3'].mean()
            diff = round(high_study - low_study, 2)
            st.markdown(
                f"<p style='color:#f0f0f0;'>Students who study more than average score <b>{diff}</b> points higher on average than those who study less.</p>",
                unsafe_allow_html=True)
    else:
        st.info("👆 Upload a CSV file to begin exploring insights about student performance.")


# -------------- PAGE 4: ABOUT -----------------
else:
    st.subheader("ℹ️ About This App")
    st.write("""
    This web app predicts whether a student is likely to pass or fail based on multiple academic parameters such as:
    - Study time per week  
    - Number of past failures  
    - Absences and prior grades  

    **Tech Stack**  
    - Streamlit for the web interface  
    - Scikit-learn model  
    - Python, NumPy, Pandas, Matplotlib  

    Built with Streamlit by Madhav Murali V*
    """)

st.markdown("</div>", unsafe_allow_html=True)
