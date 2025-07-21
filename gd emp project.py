import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Streamlit page config
st.set_page_config(page_title="💼 Employee Salary Predictor", layout="centered")
st.title("💼 Employee Salary Predictor")
st.markdown("Predict whether an employee earns **>50K** or **<=50K** using demographic and job features.")

# Step 1: CSV file path input
csv_path = st.text_input("📁 Enter the path to your CSV file", value="adult 3.csv")

if not csv_path:
    st.warning("⚠️ Please provide a valid CSV path to continue.")
    st.stop()

# Step 2: Load and preprocess dataset
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)

    # Clean missing and invalid values
    df.replace("?", np.nan, inplace=True)
    df.dropna(subset=['workclass', 'occupation', 'native-country'], inplace=True)
    df = df[~df['workclass'].isin(['Without-pay', 'Never-worked', 'Preschool'])]
    df = df[~df['education'].isin(['5th-6th', '1th-4th'])]
    df = df[(df['age'] >= 17) & (df['age'] <= 75)]

    # Fill missing numerics with median and categoricals with mode
    for col in df.select_dtypes(include='number'):
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include='object'):
        df[col].fillna(df[col].mode()[0], inplace=True)

    df.drop(columns=['education'], inplace=True, errors='ignore')

    # Encode categorical
    encoders = {}
    for col in ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    y = df['income']
    X = df.drop(columns=['income'])
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X_scaled, y, scaler, encoders, X.columns.tolist()

# Load clean data
try:
    raw_df, X, y, scaler, encoders, feature_names = load_data(csv_path)
except Exception as e:
    st.error(f"❌ Failed to load data: {e}")
    st.stop()

# Show raw data
st.subheader("🔍 Raw Data Preview")
st.markdown("#### 📄 Preview of cleaned input data used for model")
st.dataframe(raw_df.head(50), height=500, use_container_width=True)

# Step 3: Train model
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# Step 4: Sidebar input
st.sidebar.header("🧾 Enter Employee Details")
sidebar_inputs = {}
for feature in feature_names:
    if feature == 'age':
        sidebar_inputs[feature] = st.sidebar.slider("👤 Age", 17, 75, 30)
    elif feature == 'hours-per-week':
        sidebar_inputs[feature] = st.sidebar.slider("⏱ Hours per Week", 1, 60, 40)
    elif feature in encoders:
        options = list(encoders[feature].classes_)
        selected = st.sidebar.selectbox(f"{feature.capitalize()}", options)
        sidebar_inputs[feature] = encoders[feature].transform([selected])[0]
    else:
        sidebar_inputs[feature] = st.sidebar.number_input(f"{feature.capitalize()}")

# Step 5: Encode and predict
if st.sidebar.button("🔍 Predict Salary Class"):
    try:
        input_df = pd.DataFrame([sidebar_inputs])
        input_df = input_df[feature_names]
        scaled_input = scaler.transform(input_df)
        pred = model.predict(scaled_input)[0]
        st.subheader("📊 Prediction Result")
        st.success(f"💰 Predicted Salary: {'>50K' if pred == 1 else '<=50K'}")
    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")

# Step 6: Batch Prediction
st.subheader("📂 Batch Prediction from CSV")
uploaded_file = st.file_uploader("Upload a CSV file with same feature columns", type=["csv"])
if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)
        for col in encoders:
            if col in batch_df.columns:
                batch_df[col] = encoders[col].transform(batch_df[col].astype(str))
        batch_scaled = scaler.transform(batch_df[feature_names])
        preds = model.predict(batch_scaled)
        batch_df['Predicted Salary'] = ['>50K' if p == 1 else '<=50K' for p in preds]
        st.write(batch_df)
        st.download_button("📥 Download Predictions", batch_df.to_csv(index=False), "predictions.csv")
    except Exception as e:
        st.error(f"❌ Error: {e}")

# Footer
st.markdown("---")
st.caption("✨ Made with ❤️ using Streamlit and Machine Learning – by Darshil & Gattha's Assistant")