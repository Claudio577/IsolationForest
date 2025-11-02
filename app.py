# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

# --- CONFIGURAÇÕES ---
st.set_page_config(page_title="Sentinela IA de Fraude Financeira", layout="wide")
st.title("💳 Sentinela IA — Detecção de Fraudes com XGBoost")

st.markdown("""
Este aplicativo usa o modelo **XGBoost (Machine Learning Supervisionado)** 
treinado no dataset real de **fraudes de cartão de crédito (Kaggle)**.

O objetivo é classificar transações como **Fraudulentas (1)** ou **Normais (0)**.
""")

# --- CARREGAMENTO DE DADOS ---
@st.cache_data
def carregar_dados():
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    df = pd.read_csv(url)
    return df


df = carregar_dados()
st.success(f"✅ Dataset carregado com {df.shape[0]:,} transações e {df.shape[1]} variáveis.")

# --- PREPARAÇÃO DOS DADOS ---
FEATURES = [col for col in df.columns if col not in ["Class"]]
X = df[FEATURES]
y = df["Class"]

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# --- TREINAMENTO ---
@st.cache_resource
def treinar_modelo():
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        scale_pos_weight=10,  # compensa o desbalanceamento
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

model = treinar_modelo()
st.success("🎯 Modelo XGBoost treinado com sucesso!")

# --- AVALIAÇÃO ---
y_pred = model.predict(X_test)
roc = roc_auc_score(y_test, y_pred)

st.subheader("📊 Avaliação no Conjunto de Teste")
st.write(f"**AUC-ROC:** {roc:.4f}")
st.write("**Relatório de Classificação:**")
st.text(classification_report(y_test, y_pred))

# --- INTERFACE DE PREDIÇÃO ---
st.header("🔍 Analisar Nova Transação")

cols = st.columns(2)
with cols[0]:
    amount = st.number_input("Valor da Transação (Amount)", min_value=0.0, value=100.0)
    time = st.number_input("Tempo (segundos desde início do dataset)", min_value=0.0, value=50000.0)
with cols[1]:
    v_features = [st.number_input(f"V{i}", value=0.0) for i in range(1, 29)]

if st.button("Analisar Risco"):
    nova_transacao = pd.DataFrame([[time, *v_features, amount]], columns=FEATURES)
    X_novo = scaler.transform(nova_transacao)
    pred = model.predict(X_novo)[0]
    prob = model.predict_proba(X_novo)[0][1]

    if pred == 1:
        st.error(f"🚨 **Alta probabilidade de fraude!** (prob={prob:.4f})")
    else:
        st.success(f"✅ **Transação normal.** (prob={prob:.4f})")

# --- INFORMAÇÕES ---
st.markdown("---")
st.markdown("""
💡 **Dicas:**
- Coloque valores extremos em algumas variáveis V1...V28 para simular anomalias.  
- O modelo é treinado com forte desbalanceamento (492 fraudes em ~284 mil transações).  
- **A métrica AUC-ROC** mede a capacidade de distinguir fraudes vs. normais.
""")

