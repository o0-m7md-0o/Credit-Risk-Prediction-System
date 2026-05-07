# ================= IMPORTS =================
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import xgboost as xgb

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Credit Risk AI",
    page_icon="💳",
    layout="wide"
)

# ================= DARK MODE STYLE =================
st.markdown("""
<style>

.stApp{
    background-color:#0E1117;
    color:white;
}

.main-title{
    font-size:45px;
    font-weight:bold;
    color:#00F5FF;
    text-align:center;
}

.sub{
    text-align:center;
    color:#A0A0A0;
    font-size:18px;
}

.card{
    background:#161B22;
    padding:25px;
    border-radius:20px;
    box-shadow:0px 0px 15px rgba(0,255,255,0.2);
}

.stButton>button{
    width:100%;
    border-radius:10px;
    height:3em;
    background:linear-gradient(90deg,#00F5FF,#0066FF);
    color:white;
    font-size:18px;
    font-weight:bold;
    border:none;
}

</style>
""", unsafe_allow_html=True)

# ================= LOAD FILES =================
model = pickle.load(open("catboost_model.pkl", "rb"))

loan_intent_encoder = pickle.load(open("loan_intent_label_encoder.pkl", "rb"))
loan_grade_encoder = pickle.load(open("loan_grade_label_encoder.pkl", "rb"))

home_ohe = pickle.load(open("person_home_ownership_ohe.pkl", "rb"))
default_ohe = pickle.load(open("cb_person_default_on_file_ohe.pkl", "rb"))

scaler = pickle.load(open("scaler.pkl", "rb"))

# ================= HEADER =================
st.markdown(
    '<p class="main-title">💳 Credit Risk Prediction System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub">AI-Powered Loan Risk Analysis Dashboard</p>',
    unsafe_allow_html=True
)

st.write("")

# ================= LAYOUT =================
col1, col2 = st.columns([1,1])

# =========================================================
# ====================== INPUTS ============================
# =========================================================

with col1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("👤 Customer Information")

    person_age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    person_income = st.number_input(
        "Annual Income",
        min_value=1000,
        max_value=1000000,
        value=50000
    )

    person_emp_length = st.number_input(
        "Employment Length",
        min_value=0,
        max_value=50,
        value=5
    )

    person_home_ownership = st.selectbox(
        "Home Ownership",
        ['RENT', 'OWN', 'MORTGAGE', 'OTHER']
    )

    loan_intent = st.selectbox(
        "Loan Intent",
        [
            'EDUCATION',
            'MEDICAL',
            'VENTURE',
            'PERSONAL',
            'HOMEIMPROVEMENT',
            'DEBTCONSOLIDATION'
        ]
    )

    loan_grade = st.selectbox(
        "Loan Grade",
        ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    )

    loan_amnt = st.number_input(
        "Loan Amount",
        min_value=500,
        max_value=100000,
        value=10000
    )

    loan_int_rate = st.number_input(
        "Interest Rate",
        min_value=1.0,
        max_value=40.0,
        value=10.0
    )

    loan_percent_income = st.number_input(
        "Loan Percent Income",
        min_value=0.0,
        max_value=1.0,
        value=0.2
    )

    cb_person_default_on_file = st.selectbox(
        "Historical Default",
        ['Y', 'N']
    )

    cb_person_cred_hist_length = st.number_input(
        "Credit History Length",
        min_value=1,
        max_value=50,
        value=5
    )

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# ==================== PREDICTION =========================
# =========================================================

with col2:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📊 Prediction Result")

    if st.button("🚀 Predict Risk"):

        # ================= LABEL ENCODING =================
        loan_intent_encoded = loan_intent_encoder.transform([loan_intent])[0]

        loan_grade_encoded = loan_grade_encoder.transform([loan_grade])[0]

        # ================= NUMERICAL DATA =================
        numerical_data = pd.DataFrame([[
            person_age,
            person_income,
            person_emp_length,
            loan_intent_encoded,
            loan_grade_encoded,
            loan_amnt,
            loan_int_rate,
            loan_percent_income,
            cb_person_cred_hist_length
        ]], columns=[
            'person_age',
            'person_income',
            'person_emp_length',
            'loan_intent',
            'loan_grade',
            'loan_amnt',
            'loan_int_rate',
            'loan_percent_income',
            'cb_person_cred_hist_length'
        ])

        # ================= SCALING =================
        numerical_scaled = scaler.transform(numerical_data)

        # ================= ONE HOT ENCODING =================
        home_encoded = home_ohe.transform([[person_home_ownership]])

        default_encoded = default_ohe.transform([[cb_person_default_on_file]])

        # ================= FINAL INPUT =================
        final_input = np.concatenate([
            numerical_scaled,
            home_encoded,
            default_encoded
        ], axis=1)

        # ================= PREDICTION =================
        prediction = model.predict(final_input)[0]

        probability = model.predict_proba(final_input)[0]

        low_risk_prob = probability[0]
        high_risk_prob = probability[1]

        # ================= RESULT =================
        if prediction == 1:
            st.error("⚠️ HIGH RISK CUSTOMER")
        else:
            st.success("✅ LOW RISK CUSTOMER")

        # ================= GAUGE CHART =================
        st.subheader("🎯 Risk Score")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=high_risk_prob * 100,
            title={'text': "High Risk Probability"},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 40], 'color': "green"},
                    {'range': [40, 70], 'color': "orange"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'bar': {'color': "red"}
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

        # ================= BAR CHART =================
        chart_df = pd.DataFrame({
            "Category": ["Low Risk", "High Risk"],
            "Probability": [
                low_risk_prob * 100,
                high_risk_prob * 100
            ]
        })

        fig2 = px.bar(
            chart_df,
            x="Category",
            y="Probability",
            text="Probability",
            color="Category",
            color_discrete_map={
                "Low Risk": "green",
                "High Risk": "red"
            }
        )

        fig2.update_traces(
            texttemplate='%{text:.2f}%',
            textposition='outside'
        )

        fig2.update_layout(
            yaxis_title="Probability %",
            xaxis_title="Category"
        )

        st.plotly_chart(fig2, use_container_width=True)

        # ================= METRICS =================
        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "High Risk %",
                f"{high_risk_prob*100:.2f}%"
            )

        with c2:
            st.metric(
                "Prediction Date",
                datetime.now().strftime("%Y-%m-%d")
            )

    st.markdown('</div>', unsafe_allow_html=True)

# ================= FOOTER =================
st.write("")
st.markdown("---")

st.markdown(
    "<center>🔥 Developed with Machine Learning & Streamlit</center>",
    unsafe_allow_html=True
)