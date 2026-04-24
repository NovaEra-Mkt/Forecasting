import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# caminho base do projeto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# carregar dados
df = pd.read_csv("C:/Users/paulo.lopes/Desktop/Project-ML-Vendas/data/dados.csv")

meses_map = {
    "janeiro": 1, "fevereiro": 2, "marco": 3, "abril": 4,
    "maio": 5, "junho": 6, "julho": 7, "agosto": 8,
    "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12
}

df["mes_num"] = df["Mes"].str.lower().map(meses_map)

# encoder
le = LabelEncoder()
df["Rede_enc"] = le.fit_transform(df["Rede"])

# salvar encoder na pasta correta
os.makedirs(os.path.join(BASE_DIR, "model"), exist_ok=True)
joblib.dump(le, os.path.join(BASE_DIR, "model", "encoder.pkl"))

# features
X = df[["Rede_enc", "mes_num", "Ano", "Qtd Vendas (cupons)"]]
y = df["Vlr Venda"]

# modelo
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

# salvar modelo
joblib.dump(model, os.path.join(BASE_DIR, "model", "modelo.pkl"))

print("Modelo treinado e salvo com sucesso!")
print("Redes: " + str(list(le.classes_)))