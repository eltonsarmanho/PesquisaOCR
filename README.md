
# 📄 PESQUISAOCR: Avaliação Comparativa de Modelos OCR

Este repositório contém o código-fonte, dados e scripts utilizados no projeto de avaliação quantitativa e qualitativa de diferentes modelos de OCR (Reconhecimento Óptico de Caracteres).

## 🎯 Objetivo

Realizar uma análise estatística rigorosa, baseada em ground truth, para comparar os seguintes modelos OCR:

- `AWS Textract`
- `Google Gemini`
- `Google Vision`
- `PaddleOCR`
- `PyTesseract`

---

## 📁 Estrutura do Projeto

```

PESQUISAOCR/
│
├── Arquivos/                 # Contém os textos extraídos e referências
│   ├── ocr\_textos/           # Saídas dos modelos OCR
│   └── truth\_textos/         # Ground truth dos documentos
│
├── Code/
│   ├── AvaliacaoQualitativa/
│   │   ├── AvaliacaoEstatistica.py
│   │   └── AvaliadorQualidade\_OCR.py
│   ├── AvaliacaoQuantitativa/
│   │   ├── AvaliacaoMedia.py
│   │   ├── AvaliacaoQuantitativaArquivos.py
│   │   └── AvaliacaoTempo\_Acuracia.py
│   └── Util/
│       └── OCR\_Extrator\_PDF\_Matricula.py
│
├── comparativo\_modelos\_ocr.csv         # Métricas por documento e por modelo
├── tempo\_extracao\_ocr.csv              # Tempos de execução por modelo
├── correlacao\_metricas.csv             # Correlação entre CER, WER e Similaridade
├── media\_metricas\_modelos.csv          # Resumo estatístico médio por modelo
├── requirements.txt                    # Dependências do projeto
└── README.md                           # Este arquivo

````

---

## 📊 Avaliações Realizadas

- **CER** (Character Error Rate)
- **WER** (Word Error Rate)
- **Levenshtein Similarity e Distance)**
- **Tempo de processamento por documento**
- **Eficiência (acurácia/tempo)**

### 📈 Testes Estatísticos Aplicados

- `Shapiro-Wilk` (normalidade)
- `ANOVA` e `Kruskal-Wallis` (diferença entre modelos)
- `Tukey HSD` e `Dunn-Bonferroni` (post-hoc)
- Correlação entre métricas

---

## 🔧 Como Executar

1. Clone este repositório:
   ```bash
   git clone https://github.com/seuusuario/PESQUISAOCR.git
   cd PESQUISAOCR
````

2. Crie o ambiente virtual e instale as dependências:

   ```bash
   python -m venv env
   source env/bin/activate  # Linux/macOS
   env\Scripts\activate     # Windows
   pip install -r requirements.txt
   ```

3. Execute os scripts desejados a partir da pasta `Code/`.



## ✍️ Autor

**Elton Sarmanho** – UFPA Cametá – `eltonss@ufpa.br`




