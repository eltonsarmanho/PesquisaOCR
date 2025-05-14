import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import kruskal
import matplotlib.cm as cm
from math import pi

# Carregar os dados dos arquivos CSV
df_metricas_ocr = pd.read_csv('comparativo_modelos_ocr.csv')
df_tempo_ocr = pd.read_csv('tempo_extracao_ocr.csv')

# Calcular métricas médias por modelo a partir do arquivo de métricas
df_metricas_media = df_metricas_ocr.groupby('modelo').agg({
    'similaridade_levenshtein': 'mean',
    'word_error_rate': 'mean',
    'char_error_rate': 'mean'
}).reset_index()

# Calcular tempo médio por modelo a partir do arquivo de tempo
df_tempo_medio = df_tempo_ocr.groupby('metodo_ocr').agg({
    'tempo_em_segundos': 'mean',
    'numero_de_caracteres': 'mean'
}).reset_index()
df_tempo_medio = df_tempo_medio.rename(columns={'metodo_ocr': 'modelo'})

# Calcular eficiência (caracteres processados por segundo)
df_tempo_medio['eficiencia'] = df_tempo_medio['numero_de_caracteres'] / df_tempo_medio['tempo_em_segundos']

# Mesclar os dataframes
df_metricas = pd.merge(df_metricas_media, df_tempo_medio[['modelo', 'tempo_em_segundos', 'eficiencia']], 
                      on='modelo', how='inner')

# Renomear colunas para facilitar o uso
df_metricas = df_metricas.rename(columns={
    'similaridade_levenshtein': 'acuracia',
    'tempo_em_segundos': 'tempo'
})

# Exibir o DataFrame resultante
print("DataFrame de métricas consolidadas:")
print(df_metricas)

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
print("\nRanking por score final:")
print(df_metricas[["modelo", "score_final"]])

# Salvar o DataFrame de métricas consolidadas
df_metricas.to_csv("metricas_consolidadas_ocr.csv", index=False)

# Preparar dados para radar chart
df_radar = df_metricas[["modelo", "acuracia_norm", "tempo_norm", "eficiencia_norm"]].copy()
df_radar = df_radar.set_index("modelo")

# Definir cores para cada modelo
colors = plt.colormaps['tab10']
color_list = [colors(i) for i in range(len(df_radar))]

# Colocar os dados no formato certo para radar
labels = ["Acurácia", "Tempo (inverso)", "Eficiência"]
num_vars = len(labels)

# Criar gráfico radar
angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += angles[:1]  # fechar o gráfico

# Criar figura com tamanho adequado
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, polar=True)

# Adicionar linhas de grade
ax.set_rlabel_position(0)
plt.yticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.50", "0.75", "1.00"], color="grey", size=8)
plt.ylim(0, 1)

# Plotar cada modelo
for i, (idx, row) in enumerate(df_radar.iterrows()):
    valores = row.values.tolist()
    valores += valores[:1]  # Fechar o polígono
    ax.plot(angles, valores, linewidth=2, linestyle='solid', color=color_list[i], label=idx)
    ax.fill(angles, valores, color=color_list[i], alpha=0.1)

# Adicionar rótulos
plt.xticks(angles[:-1], labels, size=12)
plt.title("Radar Chart dos Modelos OCR (Normalizado)", size=15, y=1.1)

# Adicionar legenda com melhor posicionamento
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), ncol=2)

plt.tight_layout()
plt.savefig("radar_plot_ocr_modelos.png", dpi=300, bbox_inches='tight')
plt.show()

# Criar um segundo gráfico de radar com os valores brutos (não normalizados)
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, polar=True)

# Preparar dados para radar chart com valores brutos
df_radar_raw = df_metricas[["modelo", "acuracia", "tempo", "eficiencia"]].copy()
df_radar_raw = df_radar_raw.set_index("modelo")

# Inverter o tempo (menor é melhor)
max_tempo = df_radar_raw["tempo"].max()
df_radar_raw["tempo"] = max_tempo - df_radar_raw["tempo"]

# Normalizar cada métrica para o radar plot
for col in df_radar_raw.columns:
    df_radar_raw[col] = df_radar_raw[col] / df_radar_raw[col].max()

# Plotar cada modelo
for i, (idx, row) in enumerate(df_radar_raw.iterrows()):
    valores = row.values.tolist()
    valores += valores[:1]  # Fechar o polígono
    ax.plot(angles, valores, linewidth=2, linestyle='solid', color=color_list[i], label=idx)
    ax.fill(angles, valores, color=color_list[i], alpha=0.1)

# Adicionar linhas de grade
ax.set_rlabel_position(0)
plt.yticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.50", "0.75", "1.00"], color="grey", size=8)
plt.ylim(0, 1)

# Adicionar rótulos
plt.xticks(angles[:-1], ["Acurácia", "Tempo (inverso)", "Eficiência"], size=12)
plt.title("Radar Chart dos Modelos OCR (Valores Relativos)", size=15, y=1.1)

# Adicionar legenda com melhor posicionamento
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), ncol=2)

plt.tight_layout()
plt.savefig("radar_plot_ocr_modelos_raw.png", dpi=300, bbox_inches='tight')
plt.show()

# Criar um gráfico de barras para o score final
plt.figure(figsize=(10, 6))
bars = plt.bar(df_metricas["modelo"], df_metricas["score_final"], color=color_list)

# Adicionar valores nas barras
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom', fontsize=10)

plt.title("Score Final dos Modelos OCR", fontsize=15)
plt.xlabel("Modelo", fontsize=12)
plt.ylabel("Score Final", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("score_final_ocr_modelos.png", dpi=300, bbox_inches='tight')
plt.show()

# Criar um gráfico de barras para as métricas individuais
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Acurácia (similaridade Levenshtein)
axes[0].bar(df_metricas["modelo"], df_metricas["acuracia"], color=color_list)
axes[0].set_title("Acurácia (Similaridade Levenshtein)", fontsize=14)
axes[0].set_xlabel("Modelo", fontsize=12)
axes[0].set_ylabel("Similaridade (0-1)", fontsize=12)
axes[0].tick_params(axis='x', rotation=45)

# Tempo médio
axes[1].bar(df_metricas["modelo"], df_metricas["tempo"], color=color_list)
axes[1].set_title("Tempo Médio de Processamento", fontsize=14)
axes[1].set_xlabel("Modelo", fontsize=12)
axes[1].set_ylabel("Tempo (segundos)", fontsize=12)
axes[1].tick_params(axis='x', rotation=45)

# Eficiência
axes[2].bar(df_metricas["modelo"], df_metricas["eficiencia"], color=color_list)
axes[2].set_title("Eficiência (caracteres/segundo)", fontsize=14)
axes[2].set_xlabel("Modelo", fontsize=12)
axes[2].set_ylabel("Caracteres por segundo", fontsize=12)
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("metricas_individuais_ocr.png", dpi=300, bbox_inches='tight')
plt.show()
