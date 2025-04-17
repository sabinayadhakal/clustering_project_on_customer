import streamlit as st
import pickle
import numpy as np

with open(r"C:\Users\Sabinaya\PROJECTS\kmeans_model.pkl", "rb") as file:
    kmeans = pickle.load(file)

with open(r"C:\Users\Sabinaya\PROJECTS\scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

st.set_page_config(page_title="Customer Segmentation", layout="centered")

st.title("🧠 Customer Segmentation using K-Means")

st.write("""
### Enter Customer Information
We'll tell you which segment they belong to!
""")

age = st.slider("Age", 18, 70, 30)
income = st.slider("Annual Income (k$)", 10, 150, 40)
spending = st.slider("Spending Score (1-100)", 1, 100, 50)

if st.button("Predict Cluster"):
    user_data = np.array([[age, income, spending]])
    user_data_scaled = scaler.transform(user_data)

    cluster = kmeans.predict(user_data_scaled)[0]

    st.success(f"🎯 This customer belongs to **Cluster {cluster}**")

    cluster_meanings = {
        0: "💸 Young & low-income big spender",
        1: "🤑 Wealthy and generous",
        2: "😐 Wealthy but low spending",
        3: "👶 Young mid-spenders",
        4: "🧍‍♂️ Older low-value customers"
    }

    st.write("### Segment Insight:")
    st.info(cluster_meanings.get(cluster, "No info available."))

st.markdown("---")
st.caption("Made with ❤️ by Sabinaya")
