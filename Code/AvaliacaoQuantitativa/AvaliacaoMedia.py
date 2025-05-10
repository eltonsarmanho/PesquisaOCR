import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados
caminho_arquivo = "tempo_extracao_ocr.csv"
df = pd.read_csv(caminho_arquivo)


# Calcular média do tempo de processamento por método OCR
media_tempo_por_metodo = df.groupby("metodo_ocr")["tempo_em_segundos"].mean().reset_index()
print("len(media_tempo_por_metodo)", len(media_tempo_por_metodo))
print(media_tempo_por_metodo)
# Plotar gráfico de barras
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=media_tempo_por_metodo, x="metodo_ocr", y="tempo_em_segundos", hue = "metodo_ocr", palette="Set2")
for i, v in enumerate(media_tempo_por_metodo['metodo_ocr']):
    ax.bar_label(ax.containers[i], fontsize=10);
plt.title("Tempo Médio de Processamento por Método OCR")
plt.xlabel("Método OCR")
plt.ylabel("Tempo Médio (segundos)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True, axis='y')
plt.show()
