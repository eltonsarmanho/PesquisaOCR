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
from pdf2image import convert_from_path
import tempfile
from PIL import Image


OCR_PYTESSERACT= 'pytesseract'
OCR_PADDLE = 'paddleocr'
OCR_GOOGLE_VISION = 'google_vision'
OCR_GOOGLE_GEMINI = 'google_gemini'
OCR_GOOGLE_EASYOCR = 'easyocr'
OCR_AWS_TEXTRACT = 'aws_textract'
OCR_CALAMARI = 'calamari'
# Carrega variáveis de ambiente
load_dotenv(override=True)
# Função para pré-processar imagem
MODEL_PATH = os.path.join(os.path.dirname(__file__),"..", 'Real-ESRGAN', 'weights', 'RealESRGAN_x4plus_anime_6B.pth')
print(MODEL_PATH)
# Função para aplicar RealESRGAN a uma imagem numpy

from basicsr.archs.rrdbnet_arch import RRDBNet

from realesrgan.utils import RealESRGANer

from PIL import Image


model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(
        scale=4,
        model_path=MODEL_PATH,
        model=model,
        tile=64,
        tile_pad=5,
        pre_pad=2,
        half=False)

#reader = easyocr.Reader(['pt'], gpu=False)
def preprocess_image(image, aplicar_superres=False):
    """
    Pré-processa a imagem para melhorar a qualidade do OCR.
    
    Args:
        image: Imagem PIL ou numpy array
        
    Returns:
        Imagem pré-processada como numpy array
    """
    # Converte para numpy array se for imagem PIL
    image = np.array(image)
    
    if aplicar_superres:
        try:
            
            image_sr = aplicar_realesrgan(image)
        
            return image_sr
        except Exception as e:
            print(f"[BSR Warning] Super resolução falhou: {e}")
    
    # Converte para escala de cinza se for colorida
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Normalização
    norm = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(norm)
        
    gamma = 1.2
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    final = cv2.LUT(clahe_img, lut)
    return final

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
            "ficha": "Ficha",
            "matricula": "matrícula", 
            "ficha": "Ficha",
            "imovel": "imóvel",
            "area": "área",
            "numero": "número",
            "registro": "registro",
            "cartorio": "cartório",
            "escritura": "escritura",
            "publica": "pública",
            "municipio": "município",
            "localizacao": "localização",
            "proprietario": "proprietário",
            "averbacao": "averbação"
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

   
def aplicar_realesrgan(imagem_np: np.ndarray, scale: int = 2) -> np.ndarray:
    """
    Aplica super-resolução Real-ESRGAN a uma imagem em formato NumPy.

    Parâmetros:
    - imagem_np (np.ndarray): imagem de entrada em formato numpy
    - scale (int): fator de escala (2 ou 4)

    Retorna:
    - imagem_np_superres (np.ndarray): imagem processada com super-resolução
    """
    
    # Convertendo np.array (cv2 formatado em BGR) para RGB (PIL Image)
    imagem_rgb = cv2.cvtColor(imagem_np, cv2.COLOR_BGR2RGB)
    imagem_pil = Image.fromarray(imagem_rgb)
    
    # Aplicar super-resolução
    output, _ = upsampler.enhance(np.array(imagem_pil), outscale=scale)

    # Converte de volta para np.ndarray
    imagem_superres = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    return imagem_superres

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

def extrair_texto(arquivo_pdf, ocr: str = "paddleocr", is_save_txt: bool = True) -> str:
    

    caminho_pdf, arquivo_temp = preparar_arquivo_para_pdf2image(arquivo_pdf)

    try:
        imagens = convert_from_path(caminho_pdf, dpi=300)
        imagens_np = [np.array(preprocess_image(img,aplicar_superres=True)) for img in imagens]

        if ocr == "pytesseract":
            texto = extrair_texto_pytesseract_imagens(imagens_np)
        elif ocr == "paddleocr":
            texto = extrair_texto_PaddleOCR_imagens(imagens_np)
        elif ocr == "easyocr":
            texto = extrair_texto_easyocr_imagens(imagens_np)
        elif ocr == "google_vision":
            texto = extrair_texto_google_vision_imagens(imagens_np)
        elif ocr == "google_gemini":
            texto = extrair_texto_google_gemini_imagens(imagens_np)
        elif ocr == "aws_textract":
            texto = extrair_texto_aws_imagens(imagens_np)
        else:
            raise ValueError(f"OCR {ocr} não reconhecido")

        if is_save_txt:
            save_txt(texto, str(arquivo_pdf), ocr)

        return texto

    finally:
        if arquivo_temp:
            arquivo_temp.close()
            os.unlink(arquivo_temp.name)

def extrair_texto_pytesseract_imagens(imagens_np):
    custom_config = r"--oem 3 --psm 11 -l por"
    texto_completo = []
    for i, imagem_np in enumerate(imagens_np):
        numero_pagina = i + 1
        texto_pagina = pytesseract.image_to_string(imagem_np, config=custom_config)
        texto_marcado = f"-- INÍCIO PÁGINA {numero_pagina} ---\n{texto_pagina}\n--- FIM PÁGINA {numero_pagina} ---\n"
        texto_completo.append(texto_marcado)
    return limpar_texto_ocr("\n".join(texto_completo))

def extrair_texto_PaddleOCR_imagens(imagens_np):
    ocr = PaddleOCR(use_angle_cls=True, lang='pt', show_log=False)
    texto_completo = []
    for i, imagem_np in enumerate(imagens_np):
        numero_pagina = i + 1
        resultados = ocr.ocr(imagem_np, cls=True)
        textos_pagina = [texto for box, (texto, conf) in resultados[0]]
        texto_pagina = "\n".join(textos_pagina)
        texto_marcado = f"\n--- INÍCIO PÁGINA {numero_pagina} ---\n{texto_pagina}\n--- FIM PÁGINA {numero_pagina} ---\n"
        texto_completo.append(texto_marcado)
    return limpar_texto_ocr("\n".join(texto_completo))

def extrair_texto_easyocr_imagens(imagens_np):
    reader = easyocr.Reader(['pt'], gpu=False, verbose=False)
    texto_completo = []
    for i, img_np in enumerate(imagens_np):
        numero_pagina = i + 1
        resultados = reader.readtext(img_np, detail=0, decoder='greedy', batch_size=1, workers=1, paragraph=False)
        texto_pagina = "\n".join(resultados)
        texto_marcado = f"\n--- INÍCIO PÁGINA {numero_pagina} ---\n{texto_pagina}\n--- FIM PÁGINA {numero_pagina} ---\n"
        texto_completo.append(texto_marcado)
    return limpar_texto_ocr("\n".join(texto_completo))

def extrair_texto_google_vision_imagens(imagens_np):
    base64_credentials = os.getenv("GOOGLE_CLOUD_CREDENTIALS_IARA")
    credentials_dict = CredentialsEncoder.convertBase64ToJson(base64_credentials)
    credenciais = service_account.Credentials.from_service_account_info(credentials_dict)
    client = vision.ImageAnnotatorClient(credentials=credenciais)

    texto_completo = []
    for i, img_np in enumerate(imagens_np):
        numero_pagina = i + 1
        img_pil = Image.fromarray(img_np)
        image_byte_array = io.BytesIO()
        img_pil.save(image_byte_array, format='JPEG')
        image = vision.Image(content=image_byte_array.getvalue())
        image_context = vision.ImageContext(language_hints=["pt"])
        response = client.document_text_detection(image=image, image_context=image_context)
        pagina_texto = response.full_text_annotation.text
        texto_marcado = f"\n--- INÍCIO PÁGINA {numero_pagina} ---\n{pagina_texto}\n--- FIM PÁGINA {numero_pagina} ---\n"
        texto_completo.append(texto_marcado)
    return limpar_texto_ocr("\n".join(texto_completo))

def extrair_texto_google_gemini_imagens(imagens_np):
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")
    prompt = """
    Extraia todo o conteúdo textual e todas as tabelas presentes na imagem de uma matrícula imobiliária, preservando estritamente a ordem de leitura original da página.

    1. Ordem de Leitura: Processe o conteúdo seguindo rigorosamente a ordem natural de leitura, da esquerda para a direita e de cima para baixo.
    2. Blocos de Texto: Separe blocos com \n\n.
    3. Tabelas: Extraia no formato CSV (linha por linha, separada por vírgula). Inclua o cabeçalho.
    4. Formato de Saída: Combine tudo no fluxo visual.
    5. Se não houver conteúdo, retorne string vazia.
    """
    texto_completo = []
    for i, img_np in enumerate(imagens_np):
        numero_pagina = i + 1
        try:
            processed_img_pil = Image.fromarray(img_np)
            response = model.generate_content(
                [prompt, processed_img_pil],
                generation_config={"max_output_tokens": 4096, "temperature": 0.2},
            )
            response.resolve()
            page_text = response.text.strip()
            texto_completo.append(f"\n--- INÍCIO PÁGINA {numero_pagina} ---\n{page_text}\n--- FIM PÁGINA {numero_pagina} ---\n")
        except Exception as page_err:
            print(f"❌ Erro na página {numero_pagina}: {page_err}")
            texto_completo.append(f"\n--- INÍCIO PÁGINA {numero_pagina} ---\n[ERRO: {page_err}]\n--- FIM PÁGINA {numero_pagina} ---\n")
    return limpar_texto_ocr("\n".join(texto_completo))

def extrair_texto_aws_imagens(imagens_np):
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_REGION")
    textract = boto3.client(
        'textract',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region
    )
    texto_completo = []
    for i, imagem_np in enumerate(imagens_np):
        numero_pagina = i + 1
        imagem_pil = Image.fromarray(imagem_np)
        buffer = io.BytesIO()
        imagem_pil.save(buffer, format='JPEG')
        imagem_bytes = buffer.getvalue()
        try:
            response = textract.detect_document_text(Document={'Bytes': imagem_bytes})
            linhas = [item["Text"] for item in response["Blocks"] if item["BlockType"] == "LINE"]
            texto_pagina = "\n".join(linhas)
        except Exception as e:
            texto_pagina = f"[ERRO AO PROCESSAR PÁGINA: {e}]"
        texto_marcado = f"\n--- INÍCIO PÁGINA {numero_pagina} ---\n{texto_pagina}\n--- FIM PÁGINA {numero_pagina} ---\n"
        texto_completo.append(texto_marcado)
    return limpar_texto_ocr("\n".join(texto_completo))