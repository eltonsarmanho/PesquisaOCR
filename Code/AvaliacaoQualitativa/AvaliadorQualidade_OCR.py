from pathlib import Path
import pandas as pd
from jiwer import wer, cer
import Levenshtein
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from scipy.stats import shapiro
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scikit_posthocs as sp
import seaborn as sns
import matplotlib.pyplot as plt
# Caminho base
BASE_DIR = Path(__file__).parent.parent.parent
print(BASE_DIR)
OCR_DIR = BASE_DIR / "Arquivos" /"ocr_textos"
GROUND_TRUTH_DIR = BASE_DIR / "Arquivos"  / "truth_textos"
MODELOS = ["google_gemini", "google_vision", "aws_textract", "paddleocr", "pytesseract", "easyocr"]

# Lista de arquivos ground truth
arquivos_truth = list(GROUND_TRUTH_DIR.glob("*.txt"))

resultados = []

for arquivo_truth in arquivos_truth:
    print(f"Processando arquivo: {arquivo_truth}")
    nome_arquivo = arquivo_truth.name
    with open(arquivo_truth, 'r', encoding='utf-8') as f:
        texto_gt = f.read()

    for modelo in MODELOS:
        arquivo_modelo = OCR_DIR / modelo / nome_arquivo
        if not arquivo_modelo.exists():
            print(f"[AVISO] Arquivo {nome_arquivo} não encontrado para modelo {modelo}")
            continue

        with open(arquivo_modelo, 'r', encoding='utf-8') as f:
            texto_ocr = f.read()

        distancia_lev = Levenshtein.distance(texto_ocr, texto_gt)
        similaridade_lev = Levenshtein.ratio(texto_ocr, texto_gt)
        erro_palavras = wer(texto_gt, texto_ocr)
        erro_caracteres = cer(texto_gt, texto_ocr)

        resultados.append({
            "arquivo": nome_arquivo,
            "modelo": modelo,
            "distancia_levenshtein": distancia_lev,
            "similaridade_levenshtein": round(similaridade_lev, 4),
            "word_error_rate": round(erro_palavras, 4),
            "char_error_rate": round(erro_caracteres, 4)
        })

# Cria DataFrame
df = pd.DataFrame(resultados)
print(df)
df.to_csv("comparativo_modelos_ocr.csv", index=False, encoding='utf-8-sig')

# Visualizações agrupadas por modelo (média)
df_media = df.groupby("modelo").mean(numeric_only=True).reset_index()

plt.figure(figsize=(8, 5))
ax = sns.barplot(data=df_media, x="modelo", y="similaridade_levenshtein", palette="Set2")
for i, v in enumerate(df_media["similaridade_levenshtein"]):
    ax.bar_label(ax.containers[i], fontsize=10);
plt.title("Similaridade Levenshtein Média por Modelo OCR")
plt.ylabel("Similaridade (0 a 1)")
plt.xlabel("Modelo")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# Gráfico de WER
plt.figure(figsize=(8, 5))
ax = sns.barplot(data=df_media, x="modelo", y="word_error_rate", palette="Set2")
for i, v in enumerate(df_media["similaridade_levenshtein"]):
    ax.bar_label(ax.containers[i], fontsize=10);
plt.title("Word Error Rate Médio por Modelo OCR")
plt.ylabel("WER")
plt.xlabel("Modelo")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# Gráfico de CER
plt.figure(figsize=(8, 5))
ax = sns.barplot(data=df_media, x="modelo", y="char_error_rate", palette="Set2")
for i, v in enumerate(df_media["similaridade_levenshtein"]):
    ax.bar_label(ax.containers[i], fontsize=10);
plt.title("Character Error Rate Médio por Modelo OCR")
plt.ylabel("CER")
plt.xlabel("Modelo")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# Estatística Kruskal-Wallis
print("\n\nEstatística Kruskal-Wallis\n")
for modelo in df['modelo'].unique():
    stat, p = shapiro(df[df['modelo'] == modelo]['char_error_rate'])
    print(f"{modelo}: W={stat:.4f}, p={p:.4f}")

grupos = [df[df["modelo"] == modelo]["char_error_rate"] for modelo in MODELOS]
cer_anova = f_oneway(*grupos)
print(f"Teste de ANOVA para CER: H={cer_anova.statistic:.4f}, p={cer_anova.pvalue:.4f}")

stat, p = kruskal(*grupos)
print(f"Teste de Kruskal-Wallis para CER: H={stat:.4f}, p={p:.4f}")

dados_cer = pd.DataFrame({
    "aws_textract": df[df["modelo"] == MODELOS[2]]["char_error_rate"] ,
    "google_gemini": df[df["modelo"] == MODELOS[0]]["char_error_rate"],
    "google_vision": df[df["modelo"] == MODELOS[1]]["char_error_rate"],
    "paddleocr": df[df["modelo"] == MODELOS[3]]["char_error_rate"],
    "pytesseract": df[df["modelo"] == MODELOS[4]]["char_error_rate"],
})
dados_cer_long = dados_cer.melt(var_name="modelo", value_name="cer")

print("\n\nPost-Hoc Dunn\n")
posthoc_dunn = sp.posthoc_dunn(dados_cer_long, val_col='cer', group_col='modelo', p_adjust='bonferroni')
print(posthoc_dunn)

tukey_result = pairwise_tukeyhsd(endog=dados_cer_long["cer"],
                                 groups=dados_cer_long["modelo"],
                                 alpha=0.05)
print(tukey_result.summary())

corr = df[["similaridade_levenshtein", "word_error_rate", "char_error_rate"]].corr()
print(corr)
df_media["rank_similaridade"] = df_media["similaridade_levenshtein"].rank(ascending=False)
df_media["rank_wer"] = df_media["word_error_rate"].rank(ascending=True)
df_media["rank_cer"] = df_media["char_error_rate"].rank(ascending=True)
df_media["ranking_final"] = df_media[["rank_similaridade", "rank_wer", "rank_cer"]].mean(axis=1)
print(df_media)
df_media.to_csv("media_metricas_modelos.csv", index=False)
corr.to_csv("correlacao_metricas.csv")