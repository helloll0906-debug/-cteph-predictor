import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="CTEPD风险预测", page_icon="🫀", layout="wide")

@st.cache_resource
def load_model():
    model = joblib.load('model.pkl')
    imputer = joblib.load('imputer.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, imputer, scaler

st.title("🫀 CTEPD风险预测在线计算器")
st.markdown("基于XGBoost模型，评估PTE患者进展为CTEPD的风险")

st.sidebar.header("请输入患者信息")

d_dimer = st.sidebar.number_input("D-二聚体 (mg/L)", value=0.5, step=0.1)
pe_history = st.sidebar.selectbox("肺栓塞病史", [0,1], format_func=lambda x: "有" if x==1 else "无")
heart_failure = st.sidebar.selectbox("心力衰竭", [0,1], format_func=lambda x: "有" if x==1 else "无")
stroke = st.sidebar.selectbox("14天内脑卒中", [0,1], format_func=lambda x: "有" if x==1 else "无")
wbc = st.sidebar.number_input("白细胞计数 (×10^9/L)", value=7.0, step=0.5)

if st.sidebar.button("开始风险评估", type="primary"):
    model, imputer, scaler = load_model()
    
    input_df = pd.DataFrame([[d_dimer, pe_history, heart_failure, stroke, wbc]], 
                            columns=['D-dimer (mg/L)', '肺栓塞病史(0=no; 1=yes)', '心衰(0=absent; 1=present)', '14天内脑卒中(0=absent; 1=present)', 'WBC(*10^9/L)'])
    
    input_scaled = scaler.transform(imputer.transform(input_df))
    prob = model.predict_proba(input_scaled)[0, 1]
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("预测CTEPD风险概率", f"{prob:.2%}")
    with col2:
        if prob >= 0.5:
            st.error("⚠️ 高风险患者，建议进一步评估")
        else:
            st.success("✅ 低风险患者，建议常规随访")
    
    st.progress(float(prob))

st.markdown("---")
st.markdown("⚠️ 本工具仅供临床参考，不能替代医生诊断")
