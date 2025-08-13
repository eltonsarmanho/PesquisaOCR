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

# ====== NOVO: análise dos resultados do teste_sistematico_ocr.py ======
import re
from datetime import datetime
# NOVO: utilidades para relatório HTML com imagens incorporadas
import base64
from io import BytesIO

# Caminho base
BASE_DIR = Path(__file__).parent.parent.parent
print(BASE_DIR)
OCR_DIR = BASE_DIR / "Arquivos" /"ocr_textos"
GROUND_TRUTH_DIR = BASE_DIR / "Arquivos"  / "truth_textos"
RESULTADOS_TESTES_DIR = BASE_DIR / "resultados_testes"
MODELOS = ["google_gemini", "google_vision", "aws_textract", "paddleocr", "pytesseract", "easyocr"]


def _get_latest_results_files():
    """Obtém os últimos arquivos detalhados e de estatísticas do diretório resultados_testes."""
    if not RESULTADOS_TESTES_DIR.exists():
        return None, None, None

    detalhados = sorted(RESULTADOS_TESTES_DIR.glob("resultados_detalhados_*.csv"))
    estatisticas = sorted(RESULTADOS_TESTES_DIR.glob("estatisticas_*.csv"))

    if not detalhados:
        return None, None, None

    detalhado_path = detalhados[-1]
    # Tentar casar estatística com o mesmo timestamp
    m = re.search(r"(\d{8}_\d{6})", detalhado_path.name)
    timestamp = m.group(1) if m else None
    stats_match = RESULTADOS_TESTES_DIR / f"estatisticas_{timestamp}.csv" if timestamp else None
    stats_path = stats_match if stats_match and stats_match.exists() else (estatisticas[-1] if estatisticas else None)

    return detalhado_path, stats_path, timestamp


def _fig_to_data_uri(fig) -> str:
    """Converte uma figura Matplotlib em data URI base64 para incorporar no HTML."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('ascii')
    plt.close(fig)
    return f"data:image/png;base64,{b64}"


def analisar_resultados_sistematicos(mostrar_plots: bool = True):
    """Lê resultados do teste sistemático e gera gráficos e ranking por método OCR.
    - Requer arquivos CSV gerados por teste_sistematico_ocr.py em resultados_testes/
    """
    detalhado_path, stats_path, timestamp = _get_latest_results_files()
    if not detalhado_path:
        print("[INFO] Nenhum arquivo 'resultados_detalhados_*.csv' encontrado em resultados_testes/. Pulando análise sistemática.")
        return

    print(f"[INFO] Lendo resultados detalhados: {detalhado_path}")
    df_det = pd.read_csv(detalhado_path)

    # Normalizar colunas esperadas
    for col in [
        'tempo_segundos', 'num_caracteres', 'num_paginas',
        'taxa_caracteres_por_segundo', 'taxa_paginas_por_segundo',
        'similaridade_levenshtein', 'word_error_rate', 'char_error_rate'
    ]:
        if col in df_det.columns:
            df_det[col] = pd.to_numeric(df_det[col], errors='coerce')

    df_det['sucesso'] = df_det['erro'].isna() | (df_det['erro'].astype(str).str.len() == 0)

    # Agregações por método
    agrupado = (
        df_det.groupby('metodo_ocr')
        .agg(
            tempo_medio=('tempo_segundos', 'mean'),
            tempo_mediano=('tempo_segundos', 'median'),
            tempo_desvio=('tempo_segundos', 'std'),
            chars_por_seg_medio=('taxa_caracteres_por_segundo', 'mean'),
            pags_por_seg_medio=('taxa_paginas_por_segundo', 'mean'),
            similaridade_levenshtein=('similaridade_levenshtein', 'mean'),
            word_error_rate=('word_error_rate', 'mean'),
            char_error_rate=('char_error_rate', 'mean'),
            taxa_sucesso=('sucesso', 'mean'),
            n_testes=('metodo_ocr', 'count')
        )
        .reset_index()
    )

    # Ranks solicitados
    df_media = agrupado.copy()
    if not df_media.empty:
        df_media['rank_similaridade'] = df_media['similaridade_levenshtein'].rank(ascending=False, method='min')
        df_media['rank_word_error_rate'] = df_media['word_error_rate'].rank(ascending=True, method='min')
        df_media['rank_char_error_rate'] = df_media['char_error_rate'].rank(ascending=True, method='min')
        # Extra: velocidade
        df_media['rank_velocidade'] = df_media['chars_por_seg_medio'].rank(ascending=False, method='min')
        # Ranking final (média dos três solicitados)
        df_media['ranking_final'] = df_media[[
            'rank_similaridade', 'rank_word_error_rate', 'rank_char_error_rate'
        ]].mean(axis=1)

    # Saídas
    ts = timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = RESULTADOS_TESTES_DIR
    out_dir.mkdir(exist_ok=True)

    caminho_media = out_dir / f"media_metricas_modelos_sistematico_{ts}.csv"
    df_media.to_csv(caminho_media, index=False)
    print(f"[OK] Métricas médias por método salvas em: {caminho_media}")

    ranking_cols = [
        'metodo_ocr', 'rank_similaridade', 'rank_word_error_rate', 'rank_char_error_rate',
        'rank_velocidade', 'ranking_final'
    ]
    caminho_ranking = out_dir / f"ranking_modelos_sistematico_{ts}.csv"
    df_media.sort_values('ranking_final').loc[:, ranking_cols].to_csv(caminho_ranking, index=False)
    print(f"[OK] Ranking salvo em: {caminho_ranking}")

    # Correlação entre métricas
    corr = df_media[[
        'similaridade_levenshtein', 'word_error_rate', 'char_error_rate', 'chars_por_seg_medio', 'tempo_medio'
    ]].corr(numeric_only=True)
    caminho_corr = out_dir / f"correlacao_metricas_sistematico_{ts}.csv"
    corr.to_csv(caminho_corr)
    print(f"[OK] Correlação salva em: {caminho_corr}")

    # Gráficos
    plt.figure(figsize=(9, 5))
    sns.barplot(data=df_media, x='metodo_ocr', y='chars_por_seg_medio', palette='Set2')
    plt.title('Velocidade média (caracteres/segundo) por método OCR')
    plt.ylabel('chars/s (média)')
    plt.xlabel('Método')
    plt.xticks(rotation=25)
    plt.tight_layout()
    caminho_png = out_dir / f"velocidade_chars_por_seg_{ts}.png"
    plt.savefig(caminho_png, dpi=150)
    if mostrar_plots:
        plt.show()
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.barplot(data=df_media, x='metodo_ocr', y='similaridade_levenshtein', palette='Set2')
    plt.title('Similaridade Levenshtein média por método OCR')
    plt.ylabel('Similaridade (0–1)')
    plt.xlabel('Método')
    plt.xticks(rotation=25)
    plt.tight_layout()
    caminho_png = out_dir / f"similaridade_levenshtein_{ts}.png"
    plt.savefig(caminho_png, dpi=150)
    if mostrar_plots:
        plt.show()
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.barplot(data=df_media, x='metodo_ocr', y='word_error_rate', palette='Set2')
    plt.title('Word Error Rate (WER) médio por método OCR')
    plt.ylabel('WER (menor é melhor)')
    plt.xlabel('Método')
    plt.xticks(rotation=25)
    plt.tight_layout()
    caminho_png = out_dir / f"wer_medio_{ts}.png"
    plt.savefig(caminho_png, dpi=150)
    if mostrar_plots:
        plt.show()
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.barplot(data=df_media, x='metodo_ocr', y='char_error_rate', palette='Set2')
    plt.title('Character Error Rate (CER) médio por método OCR')
    plt.ylabel('CER (menor é melhor)')
    plt.xlabel('Método')
    plt.xticks(rotation=25)
    plt.tight_layout()
    caminho_png = out_dir / f"cer_medio_{ts}.png"
    plt.savefig(caminho_png, dpi=150)
    if mostrar_plots:
        plt.show()
    plt.close()

    # Boxplot de tempos
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df_det, x='metodo_ocr', y='tempo_segundos')
    plt.title('Distribuição de tempo por método OCR')
    plt.ylabel('Tempo (s)')
    plt.xlabel('Método')
    plt.xticks(rotation=25)
    plt.tight_layout()
    caminho_png = out_dir / f"tempo_boxplot_{ts}.png"
    plt.savefig(caminho_png, dpi=150)
    if mostrar_plots:
        plt.show()
    plt.close()

    # Exibe top-5 do ranking no console
    print("\nTop métodos por ranking_final (menor é melhor):")
    print(df_media.sort_values('ranking_final')[['metodo_ocr', 'ranking_final', 'rank_similaridade', 'rank_word_error_rate', 'rank_char_error_rate', 'rank_velocidade']].head(5))


# Executa a análise sistemática automaticamente se houver arquivos disponíveis
try:
    analisar_resultados_sistematicos(mostrar_plots=True)
except Exception as e:
    print(f"[AVISO] Falha na análise dos resultados sistemáticos: {e}")

# ====== FIM DO BLOCO NOVO ======

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

def gerar_relatorio_html():
    """Gera um relatório HTML completo com gráficos incorporados (inline, sem paths relativos).
    Inclui: análise sistemática (resultados_testes) e análise por MODELOS/ground truth.
    """
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = RESULTADOS_TESTES_DIR
    out_dir.mkdir(exist_ok=True)

    # ---------- Seção 1: Análise Sistemática ----------
    df_sys_det = None
    df_sys_media = None
    sys_imgs = {}
    detalhado_path, stats_path, sys_ts = _get_latest_results_files()
    if detalhado_path is not None:
        df_sys_det = pd.read_csv(detalhado_path)
        # normalização
        for col in [
            'tempo_segundos', 'num_caracteres', 'num_paginas',
            'taxa_caracteres_por_segundo', 'taxa_paginas_por_segundo',
            'similaridade_levenshtein', 'word_error_rate', 'char_error_rate'
        ]:
            if col in df_sys_det.columns:
                df_sys_det[col] = pd.to_numeric(df_sys_det[col], errors='coerce')
        df_sys_det['sucesso'] = df_sys_det['erro'].isna() | (df_sys_det['erro'].astype(str).str.len() == 0)
        df_sys_media = (
            df_sys_det.groupby('metodo_ocr')
            .agg(
                tempo_medio=('tempo_segundos', 'mean'),
                tempo_mediano=('tempo_segundos', 'median'),
                tempo_desvio=('tempo_segundos', 'std'),
                chars_por_seg_medio=('taxa_caracteres_por_segundo', 'mean'),
                pags_por_seg_medio=('taxa_paginas_por_segundo', 'mean'),
                similaridade_levenshtein=('similaridade_levenshtein', 'mean'),
                word_error_rate=('word_error_rate', 'mean'),
                char_error_rate=('char_error_rate', 'mean'),
                taxa_sucesso=('sucesso', 'mean'),
                n_testes=('metodo_ocr', 'count')
            )
            .reset_index()
        )
        # ranks
        if not df_sys_media.empty:
            df_sys_media['rank_similaridade'] = df_sys_media['similaridade_levenshtein'].rank(ascending=False, method='min')
            df_sys_media['rank_word_error_rate'] = df_sys_media['word_error_rate'].rank(ascending=True, method='min')
            df_sys_media['rank_char_error_rate'] = df_sys_media['char_error_rate'].rank(ascending=True, method='min')
            df_sys_media['rank_velocidade'] = df_sys_media['chars_por_seg_medio'].rank(ascending=False, method='min')
            df_sys_media['ranking_final'] = df_sys_media[[
                'rank_similaridade', 'rank_word_error_rate', 'rank_char_error_rate'
            ]].mean(axis=1)

        # gráficos -> data uri
        # 1) Velocidade
        fig = plt.figure(figsize=(9, 5))
        sns.barplot(data=df_sys_media, x='metodo_ocr', y='chars_por_seg_medio', palette='Set2')
        plt.title('Velocidade média (chars/s) por método OCR')
        plt.ylabel('chars/s (média)')
        plt.xlabel('Método')
        plt.xticks(rotation=25)
        sys_imgs['velocidade'] = _fig_to_data_uri(fig)

        # 2) Similaridade
        fig = plt.figure(figsize=(9, 5))
        sns.barplot(data=df_sys_media, x='metodo_ocr', y='similaridade_levenshtein', palette='Set2')
        plt.title('Similaridade Levenshtein média por método OCR')
        plt.ylabel('Similaridade (0–1)')
        plt.xlabel('Método')
        plt.xticks(rotation=25)
        sys_imgs['similaridade'] = _fig_to_data_uri(fig)

        # 3) WER
        fig = plt.figure(figsize=(9, 5))
        sns.barplot(data=df_sys_media, x='metodo_ocr', y='word_error_rate', palette='Set2')
        plt.title('Word Error Rate (WER) médio por método OCR')
        plt.ylabel('WER (menor é melhor)')
        plt.xlabel('Método')
        plt.xticks(rotation=25)
        sys_imgs['wer'] = _fig_to_data_uri(fig)

        # 4) CER
        fig = plt.figure(figsize=(9, 5))
        sns.barplot(data=df_sys_media, x='metodo_ocr', y='char_error_rate', palette='Set2')
        plt.title('Character Error Rate (CER) médio por método OCR')
        plt.ylabel('CER (menor é melhor)')
        plt.xlabel('Método')
        plt.xticks(rotation=25)
        sys_imgs['cer'] = _fig_to_data_uri(fig)

        # 5) Boxplot tempo
        fig = plt.figure(figsize=(10, 5))
        sns.boxplot(data=df_sys_det, x='metodo_ocr', y='tempo_segundos')
        plt.title('Distribuição de tempo por método OCR')
        plt.ylabel('Tempo (s)')
        plt.xlabel('Método')
        plt.xticks(rotation=25)
        sys_imgs['tempo_box'] = _fig_to_data_uri(fig)

        # 6) Heatmap correlação
        corr_sys = df_sys_media[['similaridade_levenshtein','word_error_rate','char_error_rate','chars_por_seg_medio','tempo_medio']].corr(numeric_only=True)
        fig = plt.figure(figsize=(6, 5))
        sns.heatmap(corr_sys, annot=True, cmap='vlag', fmt='.2f')
        plt.title('Correlação de métricas (sistemático)')
        sys_imgs['corr_sys'] = _fig_to_data_uri(fig)

    # ---------- Seção 2: Avaliação com arquivos OCR_Arquivos/ground truth ----------
    resultados = []
    arquivos_truth = list(GROUND_TRUTH_DIR.glob('*.txt'))
    for arquivo_truth in arquivos_truth:
        nome_arquivo = arquivo_truth.name
        with open(arquivo_truth, 'r', encoding='utf-8') as f:
            texto_gt = f.read()
        for modelo in MODELOS:
            arquivo_modelo = OCR_DIR / modelo / nome_arquivo
            if not arquivo_modelo.exists():
                continue
            with open(arquivo_modelo, 'r', encoding='utf-8') as f:
                texto_ocr = f.read()
            distancia_lev = Levenshtein.distance(texto_ocr, texto_gt)
            similaridade_lev = Levenshtein.ratio(texto_ocr, texto_gt)
            erro_palavras = wer(texto_gt, texto_ocr)
            erro_caracteres = cer(texto_gt, texto_ocr)
            resultados.append({
                'arquivo': nome_arquivo,
                'modelo': modelo,
                'distancia_levenshtein': distancia_lev,
                'similaridade_levenshtein': round(similaridade_lev, 4),
                'word_error_rate': round(erro_palavras, 4),
                'char_error_rate': round(erro_caracteres, 4)
            })
    df_truth = pd.DataFrame(resultados)
    truth_imgs = {}
    df_truth_media = None
    corr_truth = None
    shapiro_tbl = None
    anova_res = None
    if not df_truth.empty:
        df_truth_media = df_truth.groupby('modelo').mean(numeric_only=True).reset_index()
        # Gráficos para a seção
        fig = plt.figure(figsize=(8, 5))
        sns.barplot(data=df_truth_media, x='modelo', y='similaridade_levenshtein', palette='Set2')
        plt.title('Similaridade Levenshtein média por modelo (arquivos OCR)')
        plt.ylabel('Similaridade (0–1)')
        plt.xlabel('Modelo')
        plt.xticks(rotation=30)
        truth_imgs['sim'] = _fig_to_data_uri(fig)

        fig = plt.figure(figsize=(8, 5))
        sns.barplot(data=df_truth_media, x='modelo', y='word_error_rate', palette='Set2')
        plt.title('Word Error Rate médio por modelo (arquivos OCR)')
        plt.ylabel('WER (menor é melhor)')
        plt.xlabel('Modelo')
        plt.xticks(rotation=30)
        truth_imgs['wer'] = _fig_to_data_uri(fig)

        fig = plt.figure(figsize=(8, 5))
        sns.barplot(data=df_truth_media, x='modelo', y='char_error_rate', palette='Set2')
        plt.title('Character Error Rate médio por modelo (arquivos OCR)')
        plt.ylabel('CER (menor é melhor)')
        plt.xlabel('Modelo')
        plt.xticks(rotation=30)
        truth_imgs['cer'] = _fig_to_data_uri(fig)

        # Correlação (truth)
        corr_truth = df_truth[['similaridade_levenshtein','word_error_rate','char_error_rate']].corr(numeric_only=True)
        fig = plt.figure(figsize=(5.5, 4.5))
        sns.heatmap(corr_truth, annot=True, cmap='vlag', fmt='.2f')
        plt.title('Correlação de métricas (ocr_textos vs truth)')
        truth_imgs['corr_truth'] = _fig_to_data_uri(fig)

        # Estatísticas (ex.: CER)
        try:
            grupos = [df_truth[df_truth['modelo'] == m]['char_error_rate'].dropna() for m in MODELOS]
            grupos = [g for g in grupos if len(g) > 0]
            if len(grupos) >= 2:
                estat_kruskal = kruskal(*grupos)
                # ANOVA se houver ao menos 2 grupos com >= 2 observações
                grupos_ok = [g for g in grupos if len(g) >= 2]
                if len(grupos_ok) >= 2:
                    cer_anova = f_oneway(*grupos_ok)
                    anova_res = cer_anova
                # Dunn post-hoc
                dados_cer = {m: df_truth[df_truth['modelo'] == m]['char_error_rate'] for m in MODELOS}
                dados_cer = {k: v for k, v in dados_cer.items() if len(v) > 0}
                df_cer_long = pd.DataFrame({k: pd.Series(v.values) for k, v in dados_cer.items()}).melt(var_name='modelo', value_name='cer').dropna()
                posthoc_dunn = sp.posthoc_dunn(df_cer_long, val_col='cer', group_col='modelo', p_adjust='bonferroni')
                # Tukey (apenas se possível)
                tukey_html = ''
                try:
                    tukey_res = pairwise_tukeyhsd(endog=df_cer_long['cer'], groups=df_cer_long['modelo'], alpha=0.05)
                    tukey_html = tukey_res.summary().as_html()
                except Exception:
                    tukey_html = '<p>Tukey HSD não pôde ser calculado.</p>'
            else:
                estat_kruskal = None
                posthoc_dunn = None
                tukey_html = '<p>Dados insuficientes para testes estatísticos.</p>'
        except Exception:
            estat_kruskal = None
            posthoc_dunn = None
            tukey_html = '<p>Falha ao calcular testes estatísticos.</p>'

        # Shapiro-Wilk por modelo (CER)
        try:
            sh_rows = []
            for m in sorted(df_truth['modelo'].unique()):
                serie = df_truth[df_truth['modelo'] == m]['char_error_rate'].dropna()
                if len(serie) >= 3:
                    W, p = shapiro(serie)
                    sh_rows.append({'modelo': m, 'W': round(float(W), 4), 'p': round(float(p), 4), 'n': int(len(serie))})
                else:
                    sh_rows.append({'modelo': m, 'W': None, 'p': None, 'n': int(len(serie))})
            shapiro_tbl = pd.DataFrame(sh_rows)
        except Exception:
            shapiro_tbl = None
    else:
        estat_kruskal = None
        posthoc_dunn = None
        tukey_html = '<p>Sem dados para estatísticas.</p>'

    # ---------- Montagem do HTML ----------
    def df_to_html(df, highlight_top=False):
        if highlight_top and 'ranking_final' in df.columns:
            # Aplica classes CSS para destacar o top 3
            styled_df = df.copy()
            for i in range(min(3, len(styled_df))):
                if i == 0:
                    styled_df.iloc[i, styled_df.columns.get_loc('metodo_ocr')] = f"🥇 {styled_df.iloc[i]['metodo_ocr']}"
                elif i == 1:
                    styled_df.iloc[i, styled_df.columns.get_loc('metodo_ocr')] = f"🥈 {styled_df.iloc[i]['metodo_ocr']}"
                elif i == 2:
                    styled_df.iloc[i, styled_df.columns.get_loc('metodo_ocr')] = f"🥉 {styled_df.iloc[i]['metodo_ocr']}"
            return styled_df.to_html(index=False, classes='table table-striped table-ranking', border=0, escape=False, table_id='ranking-table')
        return df.to_html(index=False, classes='table table-striped', border=0, justify='center')

    estilo = """
    <style>
      body { 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
        margin: 0; padding: 20px; 
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
      }
      .container { max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
      
      h1 { 
        color: #2c3e50; 
        text-align: center; 
        font-size: 2.5em; 
        margin-bottom: 10px; 
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        background: linear-gradient(45deg, #3498db, #2ecc71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
      }
      
      h2 { 
        color: #34495e; 
        border-bottom: 3px solid #3498db; 
        padding-bottom: 10px; 
        margin-top: 40px;
        font-size: 1.8em;
      }
      
      h3 { 
        color: #2c3e50; 
        margin-top: 25px; 
        font-size: 1.4em;
        border-left: 4px solid #3498db;
        padding-left: 15px;
      }
      
      .header-info { 
        text-align: center; 
        background: #ecf0f1; 
        padding: 15px; 
        border-radius: 10px; 
        margin-bottom: 30px;
        border: 2px solid #bdc3c7;
      }
      
      .subtle { color: #7f8c8d; font-size: 14px; }
      
      .grid { 
        display: grid; 
        grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); 
        gap: 20px; 
        margin: 25px 0; 
      }
      
      .card { 
        border: 2px solid #e0e6ed; 
        border-radius: 12px; 
        padding: 20px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
      }
      
      .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
      }
      
      .card h3 {
        margin-top: 0;
        color: #2c3e50;
        text-align: center;
        font-size: 1.2em;
        border: none;
        padding: 0;
      }
      
      .table { 
        width: 100%; 
        border-collapse: collapse; 
        margin: 15px 0;
        background: white;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      }
      
      .table th { 
        background: linear-gradient(135deg, #3498db, #2980b9); 
        color: white; 
        padding: 12px 10px; 
        font-weight: 600;
        text-align: center;
        font-size: 0.9em;
      }
      
      .table td { 
        padding: 10px; 
        border-bottom: 1px solid #ecf0f1; 
        text-align: center;
        font-size: 0.9em;
      }
      
      .table tr:nth-child(even) { background-color: #f8f9fa; }
      .table tr:hover { background-color: #e3f2fd; }
      
      .table-ranking tr:nth-child(1) td:first-child { 
        background: linear-gradient(135deg, #ffd700, #ffed4e); 
        font-weight: bold; 
        color: #8b6914;
      }
      .table-ranking tr:nth-child(2) td:first-child { 
        background: linear-gradient(135deg, #c0c0c0, #e8e8e8); 
        font-weight: bold; 
        color: #555;
      }
      .table-ranking tr:nth-child(3) td:first-child { 
        background: linear-gradient(135deg, #cd7f32, #deb887); 
        font-weight: bold; 
        color: #5c3a12;
      }
      
      .badge { 
        display: inline-block; 
        padding: 4px 10px; 
        border-radius: 20px; 
        font-size: 11px; 
        font-weight: bold;
        margin-left: 8px; 
        text-transform: uppercase;
      }
      
      .badge-gold { 
        background: linear-gradient(135deg, #ffd700, #ffed4e); 
        color: #8b6914; 
        border: 2px solid #ffcc02;
        box-shadow: 0 2px 4px rgba(255, 215, 0, 0.3);
      }
      
      .badge-silver { 
        background: linear-gradient(135deg, #c0c0c0, #e8e8e8); 
        color: #555; 
        border: 2px solid #a8a8a8;
        box-shadow: 0 2px 4px rgba(192, 192, 192, 0.3);
      }
      
      .badge-bronze { 
        background: linear-gradient(135deg, #cd7f32, #deb887); 
        color: #5c3a12; 
        border: 2px solid #b87333;
        box-shadow: 0 2px 4px rgba(205, 127, 50, 0.3);
      }
      
      img { 
        max-width: 100%; 
        height: auto; 
        border-radius: 8px; 
        border: 2px solid #e0e6ed;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      }
      
      .muted { color: #95a5a6; font-size: 13px; font-style: italic; }
      
      .stats-summary {
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 20px 0;
        text-align: center;
      }
      
      .stats-summary h3 {
        margin: 0 0 15px 0;
        color: white;
        border: none;
        padding: 0;
      }
      
      .metric-box {
        display: inline-block;
        background: rgba(255,255,255,0.1);
        padding: 10px 15px;
        margin: 5px;
        border-radius: 8px;
        backdrop-filter: blur(10px);
      }
      
      .section {
        background: white;
        margin: 25px 0;
        padding: 25px;
        border-radius: 12px;
        border: 2px solid #e0e6ed;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
      }
      
      footer { 
        margin-top: 40px; 
        color: #7f8c8d; 
        font-size: 12px; 
        text-align: center;
        padding: 20px;
        border-top: 2px solid #ecf0f1;
        background: #f8f9fa;
        border-radius: 8px;
      }
      
      .no-data {
        text-align: center;
        padding: 40px;
        background: #fff3cd;
        border: 2px solid #ffeaa7;
        border-radius: 8px;
        color: #856404;
      }
    </style>
    """

    # Cabeçalho
    html_parts = [
        "<html><head><meta charset='utf-8'><title>Relatório de Avaliação OCR - Análise Comparativa</title>",
        estilo,
        "</head><body>",
        "<div class='container'>",
        f"<h1>📊 Relatório de Avaliação OCR</h1>",
        f"<div class='header-info'>",
        f"<p class='subtle'>📅 Gerado em {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}</p>",
        f"<p class='subtle'>🔬 Análise Comparativa de Performance e Qualidade de Métodos OCR</p>",
        f"</div>"
    ]

    # Seção sistemática
    html_parts.append("<div class='section'>")
    html_parts.append("<h2>🔬 1) Análise Sistemática - Testes com Múltiplas Iterações</h2>")
    if df_sys_media is None or df_sys_media.empty:
        html_parts.append("<div class='no-data'>")
        html_parts.append("<h3>⚠️ Dados Não Encontrados</h3>")
        html_parts.append("<p>Nenhum resultado encontrado em resultados_testes/. Execute o teste_sistematico_ocr.py primeiro.</p>")
        html_parts.append("</div>")
    else:
        # Resumo estatístico
        total_tests = df_sys_media['n_testes'].sum() if not df_sys_media.empty else 0
        avg_success_rate = (df_sys_media['taxa_sucesso'].mean() * 100) if not df_sys_media.empty else 0
        html_parts.append("<div class='stats-summary'>")
        html_parts.append("<h3>📈 Resumo Executivo</h3>")
        html_parts.append(f"<div class='metric-box'>Total de Testes: <strong>{total_tests}</strong></div>")
        html_parts.append(f"<div class='metric-box'>Taxa Média de Sucesso: <strong>{avg_success_rate:.1f}%</strong></div>")
        html_parts.append(f"<div class='metric-box'>Métodos Avaliados: <strong>{len(df_sys_media)}</strong></div>")
        html_parts.append("</div>")
        
        # Ranking destacado
        df_rank = df_sys_media.sort_values('ranking_final')[[
            'metodo_ocr', 'ranking_final', 'rank_similaridade', 'rank_word_error_rate', 'rank_char_error_rate', 'rank_velocidade',
            'chars_por_seg_medio', 'tempo_medio', 'similaridade_levenshtein', 'word_error_rate', 'char_error_rate', 'taxa_sucesso', 'n_testes'
        ]].reset_index(drop=True)
        
        # Formatação de colunas para melhor visualização
        df_rank_display = df_rank.copy()
        df_rank_display['ranking_final'] = df_rank_display['ranking_final'].round(2)
        df_rank_display['chars_por_seg_medio'] = df_rank_display['chars_por_seg_medio'].round(1)
        df_rank_display['tempo_medio'] = df_rank_display['tempo_medio'].round(3)
        df_rank_display['similaridade_levenshtein'] = df_rank_display['similaridade_levenshtein'].round(4)
        df_rank_display['word_error_rate'] = df_rank_display['word_error_rate'].round(4)
        df_rank_display['char_error_rate'] = df_rank_display['char_error_rate'].round(4)
        df_rank_display['taxa_sucesso'] = (df_rank_display['taxa_sucesso'] * 100).round(1)
        
        # Renomear colunas para exibição
        df_rank_display.columns = [
            'Método OCR', 'Ranking Final', 'Rank Similaridade', 'Rank WER', 'Rank CER', 'Rank Velocidade',
            'Chars/s (Média)', 'Tempo (s)', 'Similaridade', 'WER', 'CER', 'Taxa Sucesso (%)', 'N° Testes'
        ]
        
        html_parts.append("<h3>🏆 Ranking dos Métodos OCR (Menor Ranking Final = Melhor)</h3>")
        html_parts.append(df_to_html(df_rank_display, highlight_top=True))

        # Gráficos com títulos melhorados
        html_parts.append("<div class='grid'>")
        html_parts.append(f"<div class='card'><h3>⚡ Velocidade de Processamento</h3><img src='{sys_imgs.get('velocidade','')}' alt='Gráfico de Velocidade'/></div>")
        html_parts.append(f"<div class='card'><h3>🎯 Precisão (Similaridade)</h3><img src='{sys_imgs.get('similaridade','')}' alt='Gráfico de Similaridade'/></div>")
        html_parts.append(f"<div class='card'><h3>📝 Erro de Palavras (WER)</h3><img src='{sys_imgs.get('wer','')}' alt='Gráfico WER'/></div>")
        html_parts.append(f"<div class='card'><h3>🔤 Erro de Caracteres (CER)</h3><img src='{sys_imgs.get('cer','')}' alt='Gráfico CER'/></div>")
        html_parts.append(f"<div class='card'><h3>⏱️ Distribuição de Tempos</h3><img src='{sys_imgs.get('tempo_box','')}' alt='Boxplot de Tempos'/></div>")
        html_parts.append(f"<div class='card'><h3>🔗 Correlação entre Métricas</h3><img src='{sys_imgs.get('corr_sys','')}' alt='Heatmap de Correlação'/></div>")
        html_parts.append("</div>")

        html_parts.append("<h3>📊 Detalhamento de Métricas por Método</h3>")
        df_detailed = df_sys_media.copy()
        df_detailed.columns = [
            'Método OCR', 'Tempo Médio (s)', 'Tempo Mediano (s)', 'Desvio Tempo (s)',
            'Chars/s (Média)', 'Páginas/s (Média)', 'Similaridade', 'WER', 'CER',
            'Taxa Sucesso', 'N° Testes', 'Rank Similaridade', 'Rank WER', 'Rank CER',
            'Rank Velocidade', 'Ranking Final'
        ]
        html_parts.append(df_to_html(df_detailed.round(4)))
    html_parts.append("</div>")

    # Seção arquivos OCR + ground truth
    html_parts.append("<div class='section'>")
    html_parts.append("<h2>📂 2) Comparação com Ground Truth (Arquivos OCR vs Verdade Terrestre)</h2>")
    if df_truth.empty:
        html_parts.append("<div class='no-data'>")
        html_parts.append("<h3>⚠️ Dados Não Encontrados</h3>")
        html_parts.append("<p>Nenhum par de arquivo encontrado entre ocr_textos/ e truth_textos/.</p>")
        html_parts.append("</div>")
    else:
        # Resumo da avaliação
        total_comparisons = len(df_truth)
        avg_similarity = df_truth['similaridade_levenshtein'].mean()
        html_parts.append("<div class='stats-summary'>")
        html_parts.append("<h3>📋 Resumo da Avaliação</h3>")
        html_parts.append(f"<div class='metric-box'>Total de Comparações: <strong>{total_comparisons}</strong></div>")
        html_parts.append(f"<div class='metric-box'>Similaridade Média: <strong>{avg_similarity:.3f}</strong></div>")
        html_parts.append(f"<div class='metric-box'>Modelos Avaliados: <strong>{len(df_truth_media)}</strong></div>")
        html_parts.append("</div>")
        
        # Ranking da seção truth
        df_truth_rank = df_truth_media.copy()
        df_truth_rank['rank_sim'] = df_truth_rank['similaridade_levenshtein'].rank(ascending=False, method='min')
        df_truth_rank['rank_wer'] = df_truth_rank['word_error_rate'].rank(ascending=True, method='min')
        df_truth_rank['rank_cer'] = df_truth_rank['char_error_rate'].rank(ascending=True, method='min')
        df_truth_rank['ranking_combinado'] = df_truth_rank[['rank_sim', 'rank_wer', 'rank_cer']].mean(axis=1)
        df_truth_rank = df_truth_rank.sort_values('ranking_combinado')
        
        # Formatação para exibição
        df_truth_display = df_truth_rank.copy()
        df_truth_display.columns = [
            'Modelo', 'Dist. Levenshtein', 'Similaridade', 'WER', 'CER',
            'Rank Similaridade', 'Rank WER', 'Rank CER', 'Ranking Combinado'
        ]
        
        html_parts.append("<h3>🏅 Ranking por Ground Truth (Menor Ranking = Melhor)</h3>")
        html_parts.append(df_to_html(df_truth_display.round(4), highlight_top=True))
        
        # Gráficos da seção truth
        html_parts.append("<div class='grid'>")
        html_parts.append(f"<div class='card'><h3>🎯 Similaridade por Modelo</h3><img src='{truth_imgs.get('sim','')}' alt='Gráfico de Similaridade'/></div>")
        html_parts.append(f"<div class='card'><h3>📝 WER por Modelo</h3><img src='{truth_imgs.get('wer','')}' alt='Gráfico WER'/></div>")
        html_parts.append(f"<div class='card'><h3>🔤 CER por Modelo</h3><img src='{truth_imgs.get('cer','')}' alt='Gráfico CER'/></div>")
        html_parts.append(f"<div class='card'><h3>🔗 Correlação (Truth)</h3><img src='{truth_imgs.get('corr_truth','')}' alt='Heatmap de Correlação'/></div>")
        html_parts.append("</div>")
        
        # Estatísticas com formatação melhorada
        html_parts.append("<h3>📊 Análise Estatística (Character Error Rate)</h3>")
        html_parts.append("<div class='stats-summary'>")
        if estat_kruskal is not None:
            sig_level = "Significativo" if estat_kruskal.pvalue < 0.05 else "Não Significativo"
            html_parts.append(f"<div class='metric-box'>Kruskal-Wallis: H={estat_kruskal.statistic:.4f}, p={estat_kruskal.pvalue:.4f} ({sig_level})</div>")
        if anova_res is not None:
            sig_level = "Significativo" if anova_res.pvalue < 0.05 else "Não Significativo"
            html_parts.append(f"<div class='metric-box'>ANOVA: F={anova_res.statistic:.4f}, p={anova_res.pvalue:.4f} ({sig_level})</div>")
        html_parts.append("</div>")
        
        if shapiro_tbl is not None and not shapiro_tbl.empty:
            html_parts.append("<h4>🔍 Teste de Normalidade (Shapiro-Wilk) por Modelo</h4>")
            shapiro_display = shapiro_tbl.copy()
            shapiro_display.columns = ['Modelo', 'Estatística W', 'p-valor', 'N']
            html_parts.append(df_to_html(shapiro_display))
        
        if posthoc_dunn is not None:
            html_parts.append("<h4>📈 Teste Post-Hoc (Dunn com Correção Bonferroni)</h4>")
            dunn_display = posthoc_dunn.round(4).reset_index()
            html_parts.append(df_to_html(dunn_display))
        
        html_parts.append("<h4>🔬 Teste de Tukey HSD</h4>")
        html_parts.append(tukey_html)

        # Amostra de resultados detalhados
        html_parts.append("<h3>📋 Amostra de Comparações Detalhadas (Primeiros 50 registros)</h3>")
        df_sample = df_truth.head(50).copy()
        df_sample.columns = ['Arquivo', 'Modelo', 'Dist. Levenshtein', 'Similaridade', 'WER', 'CER']
        html_parts.append(df_to_html(df_sample))
    
    html_parts.append("</div>")

    # Rodapé melhorado
    html_parts.append("<footer>")
    html_parts.append("<h3>📋 Informações do Relatório</h3>")
    html_parts.append("<p><strong>🔧 Sistema:</strong> Avaliador de Qualidade OCR - Análise Comparativa de Performance</p>")
    html_parts.append("<p><strong>📊 Metodologia:</strong> Testes sistemáticos com múltiplas iterações e comparação com ground truth</p>")
    html_parts.append("<p><strong>📈 Métricas:</strong> Similaridade Levenshtein, Word Error Rate (WER), Character Error Rate (CER), Velocidade</p>")
    html_parts.append("<p><strong>🖼️ Visualização:</strong> Gráficos incorporados como data URIs (sem dependências externas)</p>")
    html_parts.append(f"<p><strong>⚡ Gerado em:</strong> {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}</p>")
    html_parts.append("</footer>")
    html_parts.append("</div></body></html>")

    html = "\n".join(html_parts)
    rel_path = out_dir / f"relatorio_ocr_{ts}.html"
    with open(rel_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"[OK] Relatório HTML salvo em: {rel_path}")


# Gera o relatório HTML ao final da execução
try:
    gerar_relatorio_html()
except Exception as e:
    print(f"[AVISO] Falha ao gerar relatório HTML: {e}")