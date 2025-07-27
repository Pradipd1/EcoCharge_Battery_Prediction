import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🔋 EcoCharge: Battery Usage Prediction")
st.write("Upload your sensor data CSV to predict battery time:")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📄 Data Preview:", df.head())

    # ✅ Load the model
    model_path = os.path.join(os.path.dirname(__file__), "..", "model", "battery_predictor.pkl")
    model = joblib.load(model_path)

    # ✅ Drop label column if present
    if 'remaining_battery_time' in df.columns:
        df = df.drop('remaining_battery_time', axis=1)

    # ✅ Match the model’s expected features
    expected_features = model.feature_names_in_
    if not all(feature in df.columns for feature in expected_features):
        missing = set(expected_features) - set(df.columns)
        st.error(f"❌ Missing required features: {missing}")
    else:
        df = df[expected_features]
        predictions = model.predict(df)

        st.write("🔋 **Predicted Remaining Battery Time (minutes):**")
        st.write(predictions)

        # 📈 Visualization Section
        st.subheader("📈 Battery Time Prediction - Line Chart (Enhanced)")

        results_df = df.copy()
        results_df["Predicted Battery Time (min)"] = predictions

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=results_df["Predicted Battery Time (min)"], marker="o", ax=ax)
        ax.set_title("Predicted Remaining Battery Time")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Time (minutes)")
        ax.grid(True)

        st.pyplot(fig)
