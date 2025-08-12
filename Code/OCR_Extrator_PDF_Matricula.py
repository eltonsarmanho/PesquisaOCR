import os
import sys
import re
import io
import tempfile

# Adiciona o diretório pai ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Imports de terceiros
import cv2
import numpy as np
import pytesseract
import boto3
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
from google.cloud import vision
from google.oauth2 import service_account
from dotenv import load_dotenv
import google.generativeai as genai

# Import condicional do PaddleOCR
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError as e:
    print(f"PaddleOCR não disponível: {e}")
    PADDLEOCR_AVAILABLE = False

# Import condicional do EasyOCR para evitar erros CUDA
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError as e:
    print(f"EasyOCR não disponível: {e}")
    EASYOCR_AVAILABLE = False
    easyocr = None

# Imports locais
from Code.Util import CredentialsEncoder

# Constantes OCR
OCR_PYTESSERACT = 'pytesseract'
OCR_PYTESSERACT_PARALLEL = 'pytesseract_parallel'
OCR_PADDLE_CPU = 'paddleocr_cpu'
OCR_PADDLE_GPU = 'paddleocr_gpu'
OCR_GOOGLE_VISION = 'google_vision'
OCR_GOOGLE_GEMINI = 'google_gemini'
OCR_EASYOCR_GPU = 'easyocr_gpu'
OCR_EASYOCR_CPU = 'easyocr_cpu'
OCR_AWS_TEXTRACT = 'aws_textract'
# Carrega variáveis de ambiente
load_dotenv(override=True)

def verificar_gpu_disponivel():
    """
    Verifica se há GPU disponível para bibliotecas de OCR.
    
    Returns:
        dict: Status de GPU para cada biblioteca
    """
    gpu_status = {
        'pytorch_cuda': False,
        'paddlepaddle_gpu': False,
        'system_info': {}
    }
    
    # Verificar PyTorch CUDA (para EasyOCR)
    try:
        import torch
        gpu_status['pytorch_cuda'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            gpu_status['system_info']['cuda_version'] = torch.version.cuda
            gpu_status['system_info']['gpu_count'] = torch.cuda.device_count()
            gpu_status['system_info']['gpu_name'] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    
    # Verificar PaddlePaddle GPU
    try:
        import paddle
        gpu_status['paddlepaddle_gpu'] = paddle.is_compiled_with_cuda()
    except ImportError:
        pass
    
    return gpu_status

#reader = easyocr.Reader(['pt'], gpu=False)
def preprocess_image(image):
    """
    Pré-processa a imagem para melhorar a qualidade do OCR.
    
    Args:
        image: Imagem PIL ou numpy array
        
    Returns:
        Imagem pré-processada como numpy array
    """
    # Converte para numpy array se for imagem PIL
    image = np.array(image)
    
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

def extrair_texto(arquivo_pdf, ocr: str = "pytesseract", is_save_txt: bool = True, use_gpu: bool = True) -> str:
    """
    Extrai texto de um arquivo PDF usando diferentes métodos OCR.
    
    Args:
        arquivo_pdf: Caminho para o arquivo PDF ou bytes do PDF
        ocr: Método OCR a ser usado ('pytesseract', 'paddleocr', 'easyocr', etc.)
        is_save_txt: Se deve salvar o texto extraído em arquivo
        use_gpu: Se deve tentar usar GPU (quando disponível). False força uso de CPU
        
    Returns:
        Texto extraído do PDF
    """

    caminho_pdf, arquivo_temp = preparar_arquivo_para_pdf2image(arquivo_pdf)

    try:
        imagens = convert_from_path(caminho_pdf, dpi=300)
        imagens_np = [np.array(preprocess_image(img)) for img in imagens]

        if ocr == "pytesseract":
            texto = extrair_texto_pytesseract_imagens(imagens_np, use_parallel=False)
        elif ocr == "pytesseract_parallel":
            texto = extrair_texto_pytesseract_imagens(imagens_np, use_parallel=True)
        elif ocr == "paddleocr_gpu":
            if not PADDLEOCR_AVAILABLE:
                raise ImportError("PaddleOCR não está disponível. Execute: pip install paddleocr")
            texto = extrair_texto_PaddleOCR_imagens(imagens_np, use_gpu=use_gpu)
        elif ocr == "paddleocr_cpu":
            if not PADDLEOCR_AVAILABLE:
                raise ImportError("PaddleOCR não está disponível. Execute: pip install paddleocr")
            texto = extrair_texto_PaddleOCR_imagens(imagens_np, use_gpu=False)       
        elif ocr == "easyocr_gpu":
            if not EASYOCR_AVAILABLE:
                raise ImportError("EasyOCR não está disponível devido a problemas com PyTorch/CUDA")
            texto = extrair_texto_easyocr_imagens(imagens_np, use_gpu=use_gpu)
        elif ocr == "easyocr_cpu":
            if not EASYOCR_AVAILABLE:
                raise ImportError("EasyOCR não está disponível devido a problemas com PyTorch/CUDA")
            texto = extrair_texto_easyocr_imagens(imagens_np, use_gpu=False)        
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

def extrair_texto_pytesseract_imagens(imagens_np, use_parallel=True):
    """
    Extrai texto usando PyTesseract com otimizações de performance.
    
    Args:
        imagens_np: Lista de imagens em formato numpy
        use_parallel: Se True, usa processamento paralelo (simulado com threading)
    
    Returns:
        Texto extraído e limpo
    """
    # Configuração otimizada do Tesseract
    custom_config = r"--oem 3 --psm 11 -l por -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ0123456789.,;:!?()[]{}+-*/%=<>@#$&_ "
    
    if use_parallel and len(imagens_np) > 1:
        # Processamento paralelo usando threading (já que PyTesseract libera GIL para I/O)
        from concurrent.futures import ThreadPoolExecutor
        import threading
        
        def processar_pagina(args):
            i, imagem_np = args
            numero_pagina = i + 1
            try:
                texto_pagina = pytesseract.image_to_string(imagem_np, config=custom_config)
                return numero_pagina, texto_pagina
            except Exception as e:
                print(f"Erro ao processar página {numero_pagina}: {e}")
                return numero_pagina, f"[ERRO AO PROCESSAR PÁGINA: {e}]"
        
        # Usar ThreadPoolExecutor para paralelizar
        with ThreadPoolExecutor(max_workers=min(4, len(imagens_np))) as executor:
            resultados = list(executor.map(processar_pagina, enumerate(imagens_np)))
        
        # Ordenar resultados por número da página
        resultados.sort(key=lambda x: x[0])
        
        texto_completo = []
        for numero_pagina, texto_pagina in resultados:
            texto_marcado = f"-- INÍCIO PÁGINA {numero_pagina} ---\n{texto_pagina}\n--- FIM PÁGINA {numero_pagina} ---\n"
            texto_completo.append(texto_marcado)
    else:
        # Processamento sequencial tradicional
        texto_completo = []
        for i, imagem_np in enumerate(imagens_np):
            numero_pagina = i + 1
            try:
                texto_pagina = pytesseract.image_to_string(imagem_np, config=custom_config)
            except Exception as e:
                print(f"Erro ao processar página {numero_pagina}: {e}")
                texto_pagina = f"[ERRO AO PROCESSAR PÁGINA: {e}]"
            
            texto_marcado = f"-- INÍCIO PÁGINA {numero_pagina} ---\n{texto_pagina}\n--- FIM PÁGINA {numero_pagina} ---\n"
            texto_completo.append(texto_marcado)
    
    return limpar_texto_ocr("\n".join(texto_completo))

def extrair_texto_PaddleOCR_imagens(imagens_np, use_gpu=True):
    """
    Extrai texto usando PaddleOCR.
    
    Args:
        imagens_np: Lista de imagens em formato numpy
        use_gpu: Se deve tentar usar GPU (True) ou forçar CPU (False)
        
    Returns:
        Texto extraído e limpo
    """
    if not PADDLEOCR_AVAILABLE:
        raise ImportError("PaddleOCR não está disponível. Execute: pip install paddleocr")
    
    try:
        if use_gpu:
            # Tentar inicializar PaddleOCR com GPU se disponível
            try:
                ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang='pt',
                    use_gpu=True,
                    show_log=False
                )
                print("✅ PaddleOCR inicializado com GPU")
            except Exception as gpu_error:
                print(f"⚠️  GPU não disponível para PaddleOCR, usando CPU: {gpu_error}")
                # Fallback para CPU
                ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang='pt',
                    use_gpu=False,
                    show_log=False
                )
        else:
            # Forçar uso de CPU
            print("🔧 PaddleOCR configurado para usar CPU (use_gpu=False)")
            ocr = PaddleOCR(
                use_angle_cls=True,
                lang='pt',
                use_gpu=False,
                show_log=False
            )
    except Exception as e:
        print(f"Erro ao inicializar PaddleOCR: {e}")
        raise ImportError(f"Não foi possível inicializar PaddleOCR: {e}")
    
    texto_completo = []
    for i, imagem_np in enumerate(imagens_np):
        numero_pagina = i + 1
        try:
            # Executar OCR
            resultados = ocr.ocr(imagem_np)

            if resultados and len(resultados) > 0 and resultados[0]:
                textos_pagina = []
                for item in resultados[0]:
                    try:
                        # PaddleOCR retorna formato: [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], (texto, confiança)]
                        # Verificar se o item tem pelo menos 2 elementos
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            # O segundo elemento contém (texto, confiança)
                            texto_info = item[1]
                            if isinstance(texto_info, (list, tuple)) and len(texto_info) >= 1:
                                texto = str(texto_info[0]).strip()
                                if texto:
                                    textos_pagina.append(texto)
                            elif isinstance(texto_info, str):
                                # Caso seja uma string diretamente
                                texto = texto_info.strip()
                                if texto:
                                    textos_pagina.append(texto)
                    except (ValueError, IndexError, TypeError) as e:
                        print(f"Erro ao processar item na página {numero_pagina}: {e}")
                        # Debug: mostrar a estrutura do item problemático
                        print(f"Item problemático: {item}")
                        continue
                
                texto_pagina = "\n".join(textos_pagina) if textos_pagina else "[PÁGINA SEM TEXTO DETECTADO]"
            else:
                texto_pagina = "[PÁGINA SEM TEXTO DETECTADO]"
                
        except Exception as e:
            print(f"Erro ao executar OCR na página {numero_pagina}: {e}")
            texto_pagina = f"[ERRO AO PROCESSAR PÁGINA: {e}]"
            
        texto_marcado = f"\n--- INÍCIO PÁGINA {numero_pagina} ---\n{texto_pagina}\n--- FIM PÁGINA {numero_pagina} ---\n"
        texto_completo.append(texto_marcado)
    
    return limpar_texto_ocr("\n".join(texto_completo))

def extrair_texto_easyocr_imagens(imagens_np, reader=None, use_gpu=True):
    """
    Extrai texto usando EasyOCR.
    
    Args:
        imagens_np: Lista de imagens em formato numpy
        reader: Reader EasyOCR pré-configurado (opcional)
        use_gpu: Se deve tentar usar GPU (True) ou forçar CPU (False)
        
    Returns:
        Texto extraído e limpo
    """
    if not EASYOCR_AVAILABLE:
        raise ImportError("EasyOCR não está disponível devido a problemas com PyTorch/CUDA")
    
    if reader is None:
        if use_gpu:
            # Tentar usar GPU se disponível, caso contrário usar CPU
            try:
                reader = easyocr.Reader(['pt'], gpu=True, verbose=False)
                print("✅ EasyOCR inicializado com GPU")
            except Exception as e:
                print(f"⚠️  GPU não disponível para EasyOCR, usando CPU: {e}")
                reader = easyocr.Reader(['pt'], gpu=False, verbose=False)
        else:
            # Forçar uso de CPU
            print("🔧 EasyOCR configurado para usar CPU (use_gpu=False)")
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

