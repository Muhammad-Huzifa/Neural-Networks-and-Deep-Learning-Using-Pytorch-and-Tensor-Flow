from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from adult_income.model import predict_records

ROOT = Path(__file__).resolve().parents[1]

def main():
    st.title("Adult Income classification")
    st.caption("An educational demonstration using the UCI Adult dataset.")
    path = ROOT / "artifacts/pipeline.joblib"
    if not path.is_file():
        st.info("Run python train.py from the project folder before using the interface.")
        st.stop()
    with st.form("features"):
        values = {
            "age": st.number_input("Age", 0, 120, 35),
            "workclass": st.text_input("Workclass", "Private"),
            "fnlwgt": st.number_input("Final survey weight", min_value=0, value=200000),
            "education": st.text_input("Education", "Bachelors"),
            "education-num": st.number_input("Education number", min_value=0, value=13),
            "marital-status": st.text_input("Marital status", "Never-married"),
            "occupation": st.text_input("Occupation", "Prof-specialty"),
            "relationship": st.text_input("Relationship", "Not-in-family"),
            "race": st.text_input("Race", "White"),
            "sex": st.text_input("Sex", "Male"),
            "capital-gain": st.number_input("Capital gain", min_value=0, value=0),
            "capital-loss": st.number_input("Capital loss", min_value=0, value=0),
            "hours-per-week": st.number_input("Hours per week", 0, 168, 40),
            "country": st.text_input("Country", "United-States"),
        }
        submitted = st.form_submit_button("Predict")
    if submitted:
        result = predict_records(joblib.load(path), pd.DataFrame([values]))[0]
        st.write("Predicted class:", result["prediction"])
        st.write("Probability of >50K:", f"{result['probability_gt_50k']:.3f}")

if __name__ == "__main__":
    main()
