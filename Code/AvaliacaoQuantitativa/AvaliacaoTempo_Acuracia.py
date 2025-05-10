
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados
caminho_arquivo = "tempo_extracao_ocr.csv"
df = pd.read_csv(caminho_arquivo)

df_comparativo = pd.read_csv("comparativo_modelos_ocr.csv")

# Calcular média do tempo de processamento por método OCR
media_tempo_por_metodo = df.groupby("metodo_ocr")["tempo_em_segundos"].mean().reset_index()

# Padronizar os nomes dos modelos para combinar os dois dataframes
df["metodo_ocr"] = df["metodo_ocr"].str.lower().str.strip()
df_comparativo["modelo"] = df_comparativo["modelo"].str.lower().str.strip()

# Calcular tempo médio por modelo
tempo_medio = df.groupby("metodo_ocr")["tempo_em_segundos"].mean().reset_index()
tempo_medio.columns = ["modelo", "tempo_medio"]

# Calcular acurácia média (1 - CER)
df_comparativo["acuracia"] = 1 - df_comparativo["char_error_rate"]
acuracia_medio = df_comparativo.groupby("modelo")["acuracia"].mean().reset_index()

# Mesclar os dois dataframes
df_tradeoff = pd.merge(tempo_medio, acuracia_medio, on="modelo")

# Plotar gráfico trade-off
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_tradeoff, x="tempo_medio", y="acuracia", hue="modelo", s=150)
plt.title("Tempo Médio vs. Acurácia Média (1 - CER)")
plt.xlabel("Tempo Médio (s)")
plt.ylabel("Acurácia Média")
plt.grid(True)
plt.tight_layout()
plt.show()
