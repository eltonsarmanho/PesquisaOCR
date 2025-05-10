import sys
import os
import time
import csv
from pathlib import Path
from pdf2image import convert_from_path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from Code import OCR_Extrator_PDF_Matricula

# Lista de documentos
documentos = [
    "2301.txt", "2720.txt", "4724.txt", "5494.txt", "5892.txt",
    "6971.txt", "7131.txt", "7430.txt", "12665.txt", "12688.txt",
    "12690.txt", "12867.txt", "12878.txt", "13123.txt", "13164.txt",
    "17071.txt", "17074.txt", "17433.txt", "17434.txt", "22793.txt",
    "62406.txt", "62451.txt", "62489.txt", "62497.txt", "65716.txt",
    "65717.txt", "65718.txt", "65771.txt", "65772.txt"
]

# Métodos OCR
metodos_ocr = [
    OCR_Extrator_PDF_Matricula.OCR_PYTESSERACT,
    OCR_Extrator_PDF_Matricula.OCR_GOOGLE_VISION,
    OCR_Extrator_PDF_Matricula.OCR_GOOGLE_GEMINI,
    OCR_Extrator_PDF_Matricula.OCR_PADDLE,
    #OCR_Extrator_PDF_Matricula.OCR_GOOGLE_EASYOCR,
    OCR_Extrator_PDF_Matricula.OCR_AWS_TEXTRACT
]

# Caminho para salvar log de tempos
csv_path = Path(__file__).parent / "tempo_extracao_ocr.csv"

# Cria o cabeçalho do CSV
with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["documento", "numero_de_caracteres", "numero_de_paginas", "metodo_ocr", "tempo_em_segundos"])

# Itera sobre os documentos e métodos
for doc in documentos:
    base = doc.removesuffix(".txt")
    pdf_file = Path(__file__).parent / "Arquivos" / "PDF" / f"{base}.pdf"

    try:
        # Conta número de páginas do PDF
        paginas = len(convert_from_path(pdf_file, dpi=72))
    except Exception as e:
        print(f"❌ Erro ao contar páginas de {pdf_file.name}: {e}")
        paginas = "ERRO"

    for metodo in metodos_ocr:
        print(f"\n📄 Documento: {base}.pdf | Método: {metodo}")
        try:
            inicio = time.time()
            texto = OCR_Extrator_PDF_Matricula.extrair_texto(pdf_file, ocr=metodo, is_save_txt=True)
            fim = time.time()
            duracao = round(fim - inicio, 2)
            num_caracteres = len(texto.strip())

            # Salva no CSV
            with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([f"{base}.pdf", num_caracteres, paginas, metodo, duracao])

            print(f"⏱️ Tempo: {duracao} s | Caracteres: {num_caracteres} | Páginas: {paginas}")
        except Exception as e:
            print(f"❌ Erro no documento {base}.pdf com método {metodo}: {e}")
