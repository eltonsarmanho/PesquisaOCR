import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from pdf2image import convert_from_path
from pathlib import Path
from Code.Util import CredentialsEncoder
from dotenv import load_dotenv
import google.generativeai as genai

import re
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from paddleocr import PaddleOCR
from google.cloud import vision
import io
from google.cloud import vision
from google.oauth2 import service_account
import easyocr  # Adicione no início, junto com os outros imports
from boto3 import client
from PIL import Image
import os
import boto3
import tempfile

OCR_PYTESSERACT= 'pytesseract'
OCR_PADDLE = 'paddleocr'
OCR_GOOGLE_VISION = 'google_vision'
OCR_GOOGLE_GEMINI = 'google_gemini'
OCR_GOOGLE_EASYOCR = 'easyocr'
OCR_AWS_TEXTRACT = 'aws_textract'
# Carrega variáveis de ambiente
load_dotenv(override=True)
# Função para pré-processar imagem

reader = easyocr.Reader(['pt'], gpu=False)
def preprocess_image(image):
    image = np.array(image)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Normalização e aumento de contraste
    norm = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    contrast = cv2.convertScaleAbs(norm, alpha=2.0, beta=0)  # Mais contraste
    
    adaptive = cv2.adaptiveThreshold(
    contrast, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
    blockSize=15, C=11)

    # Denoising (remoção de ruídos sem borrar letras)
    denoised = cv2.fastNlMeansDenoising(adaptive, h=30)

    return denoised



def preparar_arquivo_para_pdf2image(input_pdf):
    """
    Recebe um Path ou bytes e retorna o caminho do arquivo para uso no pdf2image.
    Se for bytes, salva em arquivo temporário e retorna o caminho.
    """
    if isinstance(input_pdf, (str, Path)):
        return str(input_pdf), None  # Caminho normal, sem temporário
    elif isinstance(input_pdf, bytes):
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_pdf.write(input_pdf)
        temp_pdf.flush()
        return temp_pdf.name, temp_pdf  # Retorna o caminho e o arquivo para posterior cleanup
    else:
        raise ValueError(f"Tipo de entrada não suportado: {type(input_pdf)}")


def limpar_texto_ocr(texto_ocr: str) -> str:
        """
        Corrige erros comuns no texto extraído via OCR.
        """
        texto_ocr = texto_ocr.replace('“', '').replace('”', '')#Remover aspas para nao gerar erro na captura
        # Remover espaços quebrando números tipo "4.559.0ha" → "4.559,0 ha"
        texto_ocr = re.sub(r"(\d)[\.\,](\d)[\.\,]?(?=\d?ha)", r"\1,\2", texto_ocr)

        # Corrige datas separadas por quebras de linha ou "." (ex: 06.02.1998)
        texto_ocr = re.sub(r"(\d{2})[\.](\d{2})[\.](\d{4})", r"\1/\2/\3", texto_ocr)

        # Remover múltiplas quebras de linha e unir sentenças quebradas
        texto_ocr = re.sub(r"\n+", "\n", texto_ocr)  # Junta linhas consecutivas
        texto_ocr = re.sub(r"(?<!\n)\n(?!\n)", " ", texto_ocr)  # Junta quebra de linha solta entre palavras

        # Corrigir termos OCR típicos que aparecem quebrados
        substituicoes = {
            "CPF nº €": "CPF nº ",
            "aptº": "Apt.",
            "matricula": "matrícula",
            "ficha": "Ficha"
        }

        for k, v in substituicoes.items():
            texto_ocr = texto_ocr.replace(k, v)
        
        texto_ocr = re.sub(r"(?<=\d) (?=\d{3}[,.])", ".", texto_ocr)

        # Corrige termos mal lidos
        correcoes_ocr = {
            "m2": "m²",
            "ha.": "ha",  # Corrige "hectares" com ponto
            "r$": "R$",
            "—": "-",  # travessão OCR para hífen
        }
        for erro, certo in correcoes_ocr.items():
            texto_ocr = texto_ocr.replace(erro, certo)

        # Corrige espaçamentos entre palavras com ponto colado (e.g., "Taguatinga.DF")
        texto_ocr = re.sub(r"([a-zA-Z])\.([A-Z])", r"\1. \2", texto_ocr)

        # Remove links quebrados de verificação de assinatura
        texto_ocr = re.sub(
        r"https:\/\/assinador\.registrodeimoveis\.org\.br\/validate\/[\w\-]+",
        "",
        texto_ocr)

        return texto_ocr.strip()
def save_txt(texto,documento,metodo):
    pasta_destino = Path(__file__).parent.parent /"Arquivos"/ "ocr_textos"/metodo
    pasta_destino.mkdir(parents=True, exist_ok=True)
    nome_txt = Path(documento).with_suffix(".txt").name
    caminho_txt = pasta_destino / nome_txt
    with open(caminho_txt, "w", encoding="utf-8") as f:
        f.write(texto)

def read_txt(documento,metodo):
    pasta_destino = Path(__file__).parent.parent /"Test"/"OCR"/"ocr_textos"/metodo
    documento = documento.replace(".pdf", "")
    nome_txt = Path(documento).with_suffix(".txt").name
    caminho_txt = pasta_destino / nome_txt
    with open(caminho_txt, "r", encoding="utf-8") as f:
        texto = f.read()
    return texto

def extrair_texto(arquivo_pdf, ocr: str = OCR_PADDLE, is_save_txt: bool = True)->str:
                   
        """
        Extrai texto de um arquivo PDF usando o método OCR especificado.

        **Parâmetros**  
        - **arquivo_pdf : Path, opcional**  
        Caminho para o arquivo PDF a ser processado. O padrão é `None`.  
        - **ocr : str, opcional**  
        O método OCR a ser utilizado para a extração de texto. As opções incluem:  
        `'pytesseract'`, `'paddleocr'`, `'google_vision'`, `'google_gemini'`,  
        `'easyocr'`, `'aws_textract'`. O padrão é `'paddleocr'`.  
        - **is_save_txt : bool, opcional**  
        Se `True`, o texto extraído será salvo em um arquivo de texto.  
        O padrão é `True`.

        **Retorna**  
        - **str**  
        O texto extraído do PDF.

        **Erros possíveis**  
        - **ValueError**  
        Caso o método OCR especificado não seja reconhecido.
        """
       

        if ocr == OCR_PYTESSERACT:
            texto = extrair_texto_pytesseract(arquivo_pdf)
        elif ocr == OCR_PADDLE:
            texto = extrair_texto_PaddleOCR(arquivo_pdf)
        elif ocr == OCR_GOOGLE_VISION:
            texto = extrair_texto_google_vision(arquivo_pdf)
        elif ocr == OCR_GOOGLE_GEMINI:
            texto = extrair_texto_google_gemini(arquivo_pdf)
        elif ocr == OCR_GOOGLE_EASYOCR:
            texto = extrair_texto_easyocr(arquivo_pdf)
        elif ocr == OCR_AWS_TEXTRACT:
            texto = extrair_texto_aws(arquivo_pdf)
        else:
            raise ValueError(f"OCR {ocr} não reconhecido")
        
        # Se is_save_txt for verdadeiro, salvar o texto em um arquivo.
        if is_save_txt:
            save_txt(texto, str(arquivo_pdf), ocr)
        
        return texto

def extrair_texto_pytesseract(arquivo_pdf) -> str:
    print("Convertendo PDF em texto usando pytesseract...")

    if not arquivo_pdf:
        raise ValueError("Arquivo PDF não definido.")

    # Prepara o arquivo: se for bytes, cria temporário
    caminho_pdf, arquivo_temp = preparar_arquivo_para_pdf2image(arquivo_pdf)

    try:
        print("Convertendo PDF em imagens...")
        images = convert_from_path(caminho_pdf, dpi=300)

        custom_config = r"--oem 3 --psm 11 -l por"  # LSTM + layout flexível
        texto_completo = []

        for i, image in enumerate(images):
            numero_pagina = i + 1
            texto_pagina = pytesseract.image_to_string(preprocess_image(image), config=custom_config)
            texto_marcado = f"===== INÍCIO PÁGINA {numero_pagina} =====\n{texto_pagina}\n===== FIM PÁGINA {numero_pagina} =====\n"

            texto_completo.append(texto_marcado)

        texto_final = "\n".join(texto_completo)
        if not texto_final.strip():
            raise ValueError("Nenhum texto extraído do PDF.")

        return limpar_texto_ocr(texto_final)

    except Exception as e:
        raise RuntimeError(f"Erro na extração de texto via pytesseract: {e}")

    finally:
        # Remove o arquivo temporário se tiver sido criado
        if arquivo_temp:
            arquivo_temp.close()
            os.unlink(arquivo_temp.name)

def extrair_texto_PaddleOCR(arquivo_pdf):
    print("Convertendo PDF em texto usando PaddleOCR...")

    caminho_pdf, arquivo_temp = preparar_arquivo_para_pdf2image(arquivo_pdf)
    
    try:
        imagens = convert_from_path(caminho_pdf, dpi=300)
        ocr = PaddleOCR(use_angle_cls=True, lang='pt', show_log=False)

        texto_completo = []

        for i, imagem in enumerate(imagens):
            numero_pagina = i + 1
            print(f"🔍 Processando página {numero_pagina}")
            imagem_np = preprocess_image(imagem)
            resultados = ocr.ocr(imagem_np, cls=True)
            textos_pagina = [texto for box, (texto, conf) in resultados[0]]
            texto_pagina = "\n".join(textos_pagina)
            texto_marcado = (
                f"\n--- INÍCIO PÁGINA {numero_pagina} ---\n"
                f"{texto_pagina}\n"
                f"--- FIM PÁGINA {numero_pagina} ---\n"
            )
            texto_completo.append(texto_marcado)

        texto_final = "\n".join(texto_completo)
        return limpar_texto_ocr(texto_final)
    
    finally:
        if arquivo_temp:
            arquivo_temp.close()
            os.unlink(arquivo_temp.name)


def extrair_texto_google_vision(arquivo_pdf):
    print("Convertendo PDF em texto usando Google Vision...")

    if not arquivo_pdf:
        raise ValueError("Arquivo PDF não definido.")

    # Prepara o arquivo (tanto Path quanto bytes)
    caminho_pdf, arquivo_temp = preparar_arquivo_para_pdf2image(arquivo_pdf)

    try:
        # Carrega credenciais
        base64_credentials = os.getenv("GOOGLE_CLOUD_CREDENTIALS_IARA")
        credentials_dict = CredentialsEncoder.convertBase64ToJson(base64_credentials)
        credenciais = service_account.Credentials.from_service_account_info(credentials_dict)
        client = vision.ImageAnnotatorClient(credentials=credenciais)

        # Converte PDF para imagens
        imagens = convert_from_path(caminho_pdf, dpi=300)

        texto_completo = []

        for i, img_pil in enumerate(imagens):
            page_number = i + 1

            # Pré-processar imagem
            img_pil = Image.fromarray(preprocess_image(img_pil))

            # Converter imagem para bytes
            image_byte_array = io.BytesIO()
            img_pil.save(image_byte_array, format='JPEG')
            image_bytes = image_byte_array.getvalue()

            # Criar objeto de imagem para o Vision
            image = vision.Image(content=image_bytes)
            image_context = vision.ImageContext(language_hints=["pt"])

            # Detecção de texto
            response = client.document_text_detection(image=image, image_context=image_context)
            pagina_texto = response.full_text_annotation.text

            texto_completo.append(f"\n--- INÍCIO PÁGINA {page_number} ---\n{pagina_texto}\n--- FIM PÁGINA {page_number} ---\n")
            print(f"✅ Página {page_number}/{len(imagens)} processada")

        texto_final = "\n".join(texto_completo)
        return limpar_texto_ocr(texto_final)

    except Exception as e:
        raise RuntimeError(f"Erro na extração de texto com Google Vision: {e}")

    finally:
        # Remove o arquivo temporário se tiver sido criado
        if arquivo_temp:
            arquivo_temp.close()
            os.unlink(arquivo_temp.name)

def extrair_texto_google_gemini(arquivo_pdf):
    print("Convertendo PDF em texto usando Google Gemini...")

    if not arquivo_pdf:
        raise ValueError("Arquivo PDF não definido.")

    # Prepara o arquivo
    caminho_pdf, arquivo_temp = preparar_arquivo_para_pdf2image(arquivo_pdf)

    try:
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(model_name="gemini-1.5-pro")

        images = convert_from_path(caminho_pdf, dpi=300)

        if not images:
            raise FileNotFoundError("Nenhuma página foi convertida do PDF.")

        prompt = """
        Extraia todo o conteúdo textual e todas as tabelas presentes na imagem de uma matrícula imobiliária, preservando estritamente a ordem de leitura original da página.

        1. Ordem de Leitura: Processe o conteúdo seguindo rigorosamente a ordem natural de leitura, da esquerda para a direita e de cima para baixo.

        2. Blocos de Texto: Separe blocos com \\n\\n.

        3. Tabelas: Extraia no formato CSV (linha por linha, separada por vírgula). Inclua o cabeçalho.

        4. Formato de Saída: Combine tudo no fluxo visual. 

        5. Se não houver conteúdo, retorne string vazia.
        """

        texto_completo = []

        for i, img in enumerate(images):
            page_number = i + 1
            try:
                processed_img_np = preprocess_image(img)
                processed_img_pil = Image.fromarray(processed_img_np)

                response = model.generate_content(
                    [prompt, processed_img_pil],
                    generation_config={"max_output_tokens": 4096}
                )
                response.resolve()
                page_text = response.text.strip()
                
                texto_completo.append(f"\n--- INÍCIO PÁGINA {page_number} ---\n{page_text}\n--- FIM PÁGINA {page_number} ---\n")
                print(f"✅ Página {page_number} processada com sucesso.")

            except Exception as page_err:
                print(f"❌ Erro na página {page_number}: {page_err}")
                texto_completo.append(f"\n--- INÍCIO PÁGINA {page_number} ---\n[ERRO AO PROCESSAR PÁGINA: {page_err}]\n--- FIM PÁGINA {page_number} ---\n")

        texto_final = "\n".join(texto_completo)
        return limpar_texto_ocr(texto_final)

    except Exception as e:
        print(f"❌ Erro geral durante o processamento com Gemini: {e}")
        raise

    finally:
        if arquivo_temp:
            arquivo_temp.close()
            os.unlink(arquivo_temp.name)

def extrair_texto_easyocr(arquivo_pdf):
    print("Convertendo PDF em texto usando EasyOCR...")

    if not arquivo_pdf:
        raise ValueError("Arquivo PDF não definido.")

    caminho_pdf, arquivo_temp = preparar_arquivo_para_pdf2image(arquivo_pdf)

    try:
        
        imagens = convert_from_path(caminho_pdf, dpi=300)

        texto_completo = []

        for i, img in enumerate(imagens):
            numero_pagina = i + 1
            print(f"🔍 Processando página {numero_pagina}")

            img_np = np.array(preprocess_image(img))  # Preprocessamento
            resultados = reader.readtext(img_np, detail=1)

            textos_pagina = [txt for _, txt, _ in resultados]
            texto_pagina = "\n".join(textos_pagina)

            texto_marcado = (
                f"\n--- INÍCIO PÁGINA {numero_pagina} ---\n"
                f"{texto_pagina}\n"
                f"--- FIM PÁGINA {numero_pagina} ---\n"
            )
            texto_completo.append(texto_marcado)

            print(f"✅ Página {numero_pagina} processada com sucesso.")

        texto_final = "\n".join(texto_completo)
        return limpar_texto_ocr(texto_final)

    except Exception as e:
        print(f"❌ Erro ao processar com EasyOCR: {e}")
        raise

    finally:
        if arquivo_temp:
            arquivo_temp.close()
            os.unlink(arquivo_temp.name)
    
def extrair_texto_aws(arquivo_pdf):
    print("Convertendo PDF em texto usando Amazon Textract...")

    if not arquivo_pdf:
        raise ValueError("Arquivo PDF não definido.")

    caminho_pdf, arquivo_temp = preparar_arquivo_para_pdf2image(arquivo_pdf)

    try:
        # Carrega credenciais do .env
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_REGION")

        if not (aws_access_key and aws_secret_key and aws_region):
            raise EnvironmentError("Credenciais da AWS não encontradas. Verifique o arquivo .env.")

        # Inicializa cliente Textract
        textract = boto3.client(
            'textract',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )

        imagens = convert_from_path(caminho_pdf, dpi=300)
        texto_completo = []

        for i, imagem in enumerate(imagens):
            numero_pagina = i + 1
            print(f"🔍 Processando página {numero_pagina}...")

            imagem_np = preprocess_image(imagem)
            imagem_pil = Image.fromarray(imagem_np)

            # Converte imagem para bytes
            buffer = io.BytesIO()
            imagem_pil.save(buffer, format='JPEG')
            imagem_bytes = buffer.getvalue()

            # Chamada Textract
            try:
                response = textract.detect_document_text(Document={'Bytes': imagem_bytes})
                linhas = [item["Text"] for item in response["Blocks"] if item["BlockType"] == "LINE"]
                texto_pagina = "\n".join(linhas)
            except Exception as e:
                texto_pagina = f"[ERRO AO PROCESSAR PÁGINA: {e}]"
                print(f"❌ Erro Textract na página {numero_pagina}: {e}")

            texto_marcado = (
                f"\n--- INÍCIO PÁGINA {numero_pagina} ---\n"
                f"{texto_pagina}\n"
                f"--- FIM PÁGINA {numero_pagina} ---\n"
            )
            texto_completo.append(texto_marcado)

        texto_final = "\n".join(texto_completo)
        return limpar_texto_ocr(texto_final)

    except Exception as e:
        print(f"❌ Erro geral durante o processamento com Textract: {e}")
        raise

    finally:
        if arquivo_temp:
            arquivo_temp.close()
            os.unlink(arquivo_temp.name)

def extrair_texto_estruturado_aws(pdf_path: Path) -> str:
    """
    Extrai e limpa texto OCR estruturado (FORMS, TABLES, SIGNATURES) usando Amazon Textract.
    Retorna somente texto legível, formatado e sem metadados como 'LINE:' ou '[SEM TEXTO]'.
    """
    

    textract = client(
        'textract',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION")
    )

    imagens = convert_from_path(pdf_path, dpi=300)
    texto_completo = []

    for i, imagem in enumerate(imagens):
        buffer = io.BytesIO()
        imagem_np = preprocess_image(imagem)
        imagem_pil = Image.fromarray(imagem_np)
        imagem_pil.save(buffer, format="JPEG")
        imagem_bytes = buffer.getvalue()

        response = textract.analyze_document(
            Document={'Bytes': imagem_bytes},
            FeatureTypes=["FORMS", "TABLES", "SIGNATURES"]
        )

        # Apenas blocos de texto legíveis (LINES ou CELLS)
        textos_lidos = []
        for block in response["Blocks"]:
            if block["BlockType"] == "LINE" and "Text" in block:
                textos_lidos.append(block["Text"])
            elif block["BlockType"] == "CELL" and "Text" in block:
                textos_lidos.append(block["Text"])

        texto_pagina = "\n".join(textos_lidos)
        texto_completo.append(f"{texto_pagina}\n")

    return limpar_texto_ocr("\n".join(texto_completo))
