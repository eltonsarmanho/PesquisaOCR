import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import kruskal

# Carregar os dados consolidados com métricas e tempos médios
df_metricas = pd.DataFrame({
    "modelo": ["aws_textract", "google_gemini", "google_vision", "paddleocr", "pytesseract"],
    "acuracia": [0.856738, 0.853421, 0.840128, 0.778959, 0.786555],
    "tempo": [50.71, 88.52, 43.28, 32.16, 34.58],
    "eficiencia": [0.01438, 0.00829, 0.01941, 0.02423, 0.02274]
})

# Calcular os rankings normalizados (Z-score ou rank inverso)
scaler = MinMaxScaler()

# Acurácia e eficiência são melhores quanto MAIOR
df_metricas[["acuracia_norm", "eficiencia_norm"]] = scaler.fit_transform(
    df_metricas[["acuracia", "eficiencia"]])

# Tempo é melhor quanto MENOR → inverter antes de normalizar
df_metricas["tempo_invertido"] = df_metricas["tempo"].max() - df_metricas["tempo"]
df_metricas["tempo_norm"] = scaler.fit_transform(df_metricas[["tempo_invertido"]])

# Score final ponderado
df_metricas["score_final"] = (
    0.4 * df_metricas["acuracia_norm"] +
    0.3 * df_metricas["tempo_norm"] +
    0.3 * df_metricas["eficiencia_norm"]
)

# Ordenar por melhor score
df_metricas = df_metricas.sort_values("score_final", ascending=False).reset_index(drop=True)
df_metricas[["modelo", "score_final"]]


from math import pi

# Preparar dados para radar chart
df_radar = df_metricas[["modelo", "acuracia_norm", "tempo_norm", "eficiencia_norm"]].copy()
df_radar = df_radar.set_index("modelo")

# Adicionar coluna de fechamento do gráfico (repete a primeira)
df_radar["categoria"] = df_radar.index
df_radar = df_radar.reset_index(drop=True)

# Colocar os dados no formato certo para radar
labels = ["Acurácia", "Tempo (inverso)", "Eficiência"]
num_vars = len(labels)

# Criar gráfico radar
angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += angles[:1]  # fechar o gráfico

plt.figure(figsize=(8, 8))
for i in range(len(df_radar)):
    valores = df_radar.iloc[i, :-1].values.tolist()
    valores += valores[:1]
    plt.polar(angles, valores, label=df_radar.loc[i, "categoria"])

plt.xticks(angles[:-1], labels)
plt.title("Radar Chart dos Modelos OCR (Normalizado)")
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.show()