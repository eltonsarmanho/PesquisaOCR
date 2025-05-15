import numpy as np
import cv2
import sys
import os
from matplotlib import pyplot as plt
from pathlib import Path
from pdf2image import convert_from_path
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Lista de documentos
documentos = [
    "2301.txt", "2720.txt", "4724.txt", "5494.txt", "5892.txt",
    "6971.txt", "7131.txt", "7430.txt", "12665.txt", "12688.txt",
    "12690.txt", "12867.txt", "12878.txt", "13123.txt", "13164.txt",
    "17071.txt", "17074.txt", "17433.txt", "17434.txt", "22793.txt",
    "62406.txt", "62451.txt", "62489.txt", "62497.txt", "65716.txt",
    "65717.txt", "65718.txt", "65771.txt", "65772.txt"
]

def show_image(title, img, cmap='gray'):
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

def preprocess_image(image, quality_level='low'):
    """
    Pré-processa a imagem para melhorar a qualidade do OCR.
    
    Args:
        image: Imagem PIL ou numpy array
        quality_level: Nível de qualidade do pré-processamento ('low', 'normal', 'high')
        
    Returns:
        Imagem pré-processada como numpy array
    """
    # Converte para numpy array se for imagem PIL
    image = np.array(image)
    show_image("Original", image, cmap='gray' if len(image.shape) == 2 else None)
    
    # Converte para escala de cinza se for colorida
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    compare_with_original(image, gray, "Cinza")
    #show_image("Cinza", gray)

    # Normalização
    norm = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    #show_image("Normalizado", norm) 
    if quality_level == 'low':
        compare_with_original(image, norm, "Normalizado")
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(norm)
        #compare_with_original(image, clahe_img, "createCLAHE")
        
        gamma = 1.2
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
        final = cv2.LUT(clahe_img, lut)
        compare_with_original(image, final, "final")
        return final
    
    elif quality_level == 'normal':
        # Aumenta contraste
        contrast = cv2.convertScaleAbs(norm, alpha=2.0, beta=0)
        show_image("Contraste aumentado", contrast)

        # Limiarização adaptativa
        adaptive = cv2.adaptiveThreshold(
            contrast, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            blockSize=15, C=11)
        show_image("Limiar adaptativo", adaptive)

        # Redução de ruído
        denoised = cv2.fastNlMeansDenoising(adaptive, h=30)
        show_image("Redução de ruído", denoised)

        return denoised
    elif quality_level == 'high':
        # Processamento intensivo para documentos de baixa qualidade
        # Aumento de contraste
        contrast = cv2.convertScaleAbs(norm, alpha=2.5, beta=10)
        show_image("Contraste aumentado", contrast)
        # Equalização de histograma para melhorar o contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        equalized = clahe.apply(contrast)
        show_image("Equalização", equalized)
        
        # Limiarização adaptativa com parâmetros mais agressivos
        adaptive = cv2.adaptiveThreshold(
            equalized, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            blockSize=11, C=9)
        show_image("Limiar adaptativo", adaptive)
            
        # Operações morfológicas para limpar ruído
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel)
        
        # Redução de ruído avançada
        denoised = cv2.fastNlMeansDenoising(opening, h=27)
        show_image("Limpeza de ruído", adaptive)
        return denoised


def compare_with_original(original, processed, etapa, cmap='gray'):
    """
    Exibe lado a lado a imagem original e a processada para comparação.
    
    Args:
        original: Imagem original (numpy array)
        processed: Imagem após a etapa de processamento
        etapa: Nome da etapa (string)
        cmap: Colormap para exibição
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray' if len(original.shape) == 2 else None)
    plt.title("Original")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(processed, cmap=cmap)
    plt.title(etapa)
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

for doc in ["2301.txt"]:
    base = doc.removesuffix(".txt")
    pdf_file = Path(__file__).parent / "Arquivos" / "PDF" / f"{base}.pdf"
    preprocess_image(convert_from_path(pdf_file, dpi=72)[0], quality_level='low')