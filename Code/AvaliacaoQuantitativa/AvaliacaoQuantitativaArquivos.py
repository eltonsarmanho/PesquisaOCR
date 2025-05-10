# Geração de gráfico após a execução
import pandas as pd
import matplotlib.pyplot as plt

# Lê os dados do CSV
df = pd.read_csv('tempo_extracao_ocr.csv')

# Converte para string o nome do documento (remover ".pdf" se desejar)
df["documento"] = df["documento"].astype(str)

# Configura a figura
plt.figure(figsize=(14, 6))
metodos_unicos = df["metodo_ocr"].unique()

# Gera barras agrupadas por documento
largura = 0.13
documentos = df["documento"].unique()
x = range(len(documentos))

for i, metodo in enumerate(metodos_unicos):
    tempos = []
    textos = []
    for doc in documentos:
        linha = df[(df["documento"] == doc) & (df["metodo_ocr"] == metodo)]
        tempo = linha["tempo_em_segundos"].values[0] if not linha.empty else 0
        chars = linha["numero_de_caracteres"].values[0] if not linha.empty else 0
        tempos.append(tempo)
        textos.append(chars)

    pos = [p + i * largura for p in x]
    plt.bar(pos, tempos, width=largura, label=metodo)
    # Adiciona texto do número de caracteres acima das barras
    for p, t in zip(pos, textos):
        plt.text(p, max(0.1, tempo), str(t), ha='center', va='bottom', fontsize=7, rotation=90)

# Ajustes do gráfico
plt.xticks([p + largura * (len(metodos_unicos)/2 - 0.5) for p in x], documentos, rotation=90)
plt.ylabel("Tempo (s)")
plt.title("Tempo de extração OCR por documento e método")
plt.legend()
plt.tight_layout()

# Salva gráfico como imagem
#grafico_path = Path(__file__).parent / "grafico_ocr_tempos.png"
#plt.savefig(grafico_path, dpi=300)
plt.show()

#print(f"\n📊 Gráfico salvo em: {grafico_path}")
