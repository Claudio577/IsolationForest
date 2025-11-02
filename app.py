import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# --- 1. CONFIGURAÇÕES E INTRODUÇÃO ---

st.set_page_config(page_title="Sentinela IA de Fraude Logística", layout="wide")
st.title("🛡️ Sentinela IA: Detecção de Anomalias Logísticas")
st.markdown("""
Este simulador usa o algoritmo **Isolation Forest (Machine Learning Não Supervisionado)** para aprender o 
padrão de "normalidade" dos logs logísticos e detectar transações que representam alto risco de fraude.
""")

# --- 2. GERAÇÃO DE DADOS DE TREINAMENTO (Simulação de um Banco de Dados Real) ---

@st.cache_data
def gerar_dados_normais(n_samples=5000):
    """Gera um DataFrame com logs logísticos normais para treinamento."""
    np.random.seed(42)
    
    # Simula dados normais
    data = {
        # 1. Distância (km): Média 500km, desvio 150km
        'Distancia_Km': np.abs(np.random.normal(500, 150, n_samples)),
        # 2. Tempo de Viagem (horas): Média 8h (com ruído), desvio 2h
        'Tempo_Viagem_Hrs': np.abs(np.random.normal(8 + (np.random.rand(n_samples) * 0.1), 2, n_samples)),
        # 3. Peso Declarado (kg): Média 200kg, desvio 50kg
        'Peso_Declarado_Kg': np.abs(np.random.normal(200, 50, n_samples)),
        # 4. Valor da Carga (R$ 1000s): Média 50, desvio 15
        'Valor_Carga_kR': np.abs(np.random.normal(50, 15, n_samples)),
    }
    df = pd.DataFrame(data)
    
    # Adiciona algumas colunas categóricas para realismo
    centros = ['SP_CD', 'RJ_CD', 'MG_CD', 'PR_CD']
    df['Origem'] = np.random.choice(centros, n_samples)
    df['Destino'] = np.random.choice([c for c in centros if c != df['Origem'].iloc[0]], n_samples)
    
    return df

# Gera os dados de treino
df_treino = gerar_dados_normais()

# --- 3. PRÉ-PROCESSAMENTO E TREINAMENTO DO MODELO ---

# Usamos apenas as colunas numéricas para o Isolation Forest
FEATURES = ['Distancia_Km', 'Tempo_Viagem_Hrs', 'Peso_Declarado_Kg', 'Valor_Carga_kR']

# Escalonamento (normalização) dos dados é crucial para ML
scaler = StandardScaler()
X_treino_scaled = scaler.fit_transform(df_treino[FEATURES])

@st.cache_resource
def treinar_modelo(X_scaled):
    """Treina o modelo Isolation Forest."""
    # O Isolation Forest é ótimo para anomalias:
    # `contamination` estima a proporção de anomalias esperadas (aqui, 1% de fraude)
    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(X_scaled)
    return model

# Treina o modelo uma única vez
model_if = treinar_modelo(X_treino_scaled)

st.success(f"✅ Modelo Isolation Forest Treinado com sucesso em {len(df_treino)} logs normais!")

# --- 4. INTERFACE DE TESTE E CLASSIFICAÇÃO ---

st.header("🔍 Inserir Novo Log para Análise de Risco")
st.markdown("Insira os parâmetros de um novo log de transporte para classificá-lo.")

col1, col2 = st.columns(2)

with col1:
    distancia = st.number_input("Distância Percorrida (Km)", value=500.0, min_value=1.0)
    tempo = st.number_input("Tempo de Viagem (Horas)", value=8.0, min_value=0.0)

with col2:
    peso = st.number_input("Peso Declarado (Kg)", value=200.0, min_value=0.1)
    valor = st.number_input("Valor da Carga (kR - Milhares de Reais)", value=50.0, min_value=0.1)

if st.button("Analisar Risco de Fraude", type="primary"):
    
    # 1. Coleta os dados de entrada
    novo_log = pd.DataFrame([[distancia, tempo, peso, valor]], columns=FEATURES)
    
    # 2. Pré-processamento: usa o mesmo scaler treinado
    X_novo_scaled = scaler.transform(novo_log)
    
    # 3. Predição (retorna -1 para anomalia/fraude e 1 para normal)
    risco_predito = model_if.predict(X_novo_scaled)
    score_anomalia = model_if.decision_function(X_novo_scaled) # Quão longe do normal o ponto está
    
    st.subheader("Resultado da Sentinela IA")
    
    if risco_predito[0] == -1:
        st.error(f"""
        # 🚨 ALERTA DE ALTO RISCO DE FRAUDE!
        O log inserido **não corresponde** ao padrão de normalidade da rede.
        O score de anomalia é **{score_anomalia[0]:.4f}**.
        """)
        st.balloons()
    else:
        st.success(f"""
        # ✅ Log Classificado como Normal.
        O log está dentro do padrão esperado de transações logísticas. 
        O score de anomalia é **{score_anomalia[0]:.4f}**.
        """)

st.markdown("---")
st.subheader("💡 Dicas de Teste para Fraude:")
st.markdown("""
Tente inserir valores que seriam impossíveis na vida real:
- **Tempo de Viagem muito baixo:** `0.01` horas para uma `Distância de 500 Km` (anomalia de velocidade).
- **Peso muito alto:** `5000 Kg` para um `Valor de Carga de 1 kR` (anomalia de valor/peso).
""")

# Opcional: Mostrar os dados de treino (para debug)
# st.dataframe(df_treino.head())
