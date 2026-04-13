#Marcus Phablo Pereira de Oliveira
import pandas as pd
import joblib
from imblearn.under_sampling import RandomUnderSampler

# Carrega os dados (ajuste o nome do arquivo)
df = pd.read_csv('data set.csv', sep=",", low_memory=False)

# Limpeza rápida 
df.columns = df.columns.str.strip()
df = df.dropna(subset=['CLASSI_FIN'])
y = df['CLASSI_FIN']
X = df.drop(columns=['CLASSI_FIN']).apply(pd.to_numeric, errors='coerce').fillna(0)

# 2. Configurar o Undersampling
# sampling_strategy='not minority' -> reduz todas as classes até o tamanho da menor
# Para um número fixo, pode passar um dicionário.
rus = RandomUnderSampler(random_state=42, sampling_strategy='not minority')

print("Original class distribution:", y.value_counts().to_dict())

# 3. Aplicação do balanceamento
X_res, y_res = rus.fit_resample(X, y)

print("Balanced class distribution:", pd.Series(y_res).value_counts().to_dict())

# 4. aplicação o Scaler em Modelo já salvou
scaler = joblib.load('scaler_treinado.joblib')
modelo = joblib.load('modelo_dengue_chiku.joblib')
model_columns = joblib.load('model_columns.joblib')

# Garantir que as colunas estão na ordem certa
X_res_df = pd.DataFrame(X_res, columns=X.columns).reindex(columns=model_columns).fillna(0)

# Transformar e Predizer
X_scaled = scaler.transform(X_res_df)
previsoes = modelo.predict(X_scaled)

print("\nPredição concluída nos dados balanceados!")