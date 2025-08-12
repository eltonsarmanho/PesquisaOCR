#!/usr/bin/env python3
"""
Script de teste para comparar performance de OCR com e sem GPU.
"""

import time
import sys
import os
from pathlib import Path

# Adiciona o diretório do projeto ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Code import OCR_Extrator_PDF_Matricula as OCR

def testar_gpu_status():
    """Testa e exibe o status de GPU disponível."""
    print("=== VERIFICAÇÃO DE GPU ===")
    gpu_status = OCR.verificar_gpu_disponivel()
    
    print(f"PyTorch CUDA disponível: {gpu_status['pytorch_cuda']}")
    print(f"PaddlePaddle GPU disponível: {gpu_status['paddlepaddle_gpu']}")
    
    if gpu_status['system_info']:
        print("\nInformações do Sistema:")
        for key, value in gpu_status['system_info'].items():
            print(f"  {key}: {value}")
    
    return gpu_status

def benchmark_ocr_methods(arquivo_teste="Arquivos/PDF/2301.pdf", test_both_gpu_cpu=True):
    """
    Testa diferentes métodos OCR e mede o tempo de execução.
    
    Args:
        arquivo_teste: Caminho para o arquivo PDF de teste
        test_both_gpu_cpu: Se deve testar tanto GPU quanto CPU para métodos compatíveis
    """
    arquivo_path = Path(__file__).parent / arquivo_teste
    
    if not arquivo_path.exists():
        print(f"❌ Arquivo de teste não encontrado: {arquivo_path}")
        return
    
    print(f"\n=== BENCHMARK OCR ===")
    print(f"Arquivo de teste: {arquivo_path.name}")
    print(f"Teste GPU/CPU: {'Sim' if test_both_gpu_cpu else 'Não'}")
    
    # Métodos OCR básicos (sempre CPU)
    metodos_teste = {
        "PyTesseract (CPU)": ("pytesseract", False),
        "PyTesseract Paralelo (CPU)": ("pytesseract_parallel", False),
    }
    
    # Verificar status da GPU
    gpu_status = OCR.verificar_gpu_disponivel()
    
    # Adicionar métodos EasyOCR
    if OCR.EASYOCR_AVAILABLE:
        if test_both_gpu_cpu:
            metodos_teste["EasyOCR (CPU Forçado)"] = ("easyocr_cpu", False)
            if gpu_status['pytorch_cuda']:
                metodos_teste["EasyOCR (GPU Forçado)"] = ("easyocr_cuda", True)
                metodos_teste["EasyOCR (Auto GPU/CPU)"] = ("easyocr", True)
            else:
                metodos_teste["EasyOCR (Auto - sem GPU)"] = ("easyocr", True)
        else:
            metodos_teste["EasyOCR (Padrão)"] = ("easyocr", True)
    
    # Adicionar métodos PaddleOCR
    if OCR.PADDLEOCR_AVAILABLE:
        if test_both_gpu_cpu:
            metodos_teste["PaddleOCR (CPU Forçado)"] = ("paddleocr_cpu", False)
            if gpu_status['paddlepaddle_gpu']:
                metodos_teste["PaddleOCR (GPU Forçado)"] = ("paddleocr_gpu", True)
                metodos_teste["PaddleOCR (Auto GPU/CPU)"] = ("paddleocr", True)
            else:
                metodos_teste["PaddleOCR (Auto - sem GPU)"] = ("paddleocr", True)
        else:
            metodos_teste["PaddleOCR (Padrão)"] = ("paddleocr", True)
    
    resultados = {}
    
    for nome_metodo, (codigo_metodo, use_gpu_default) in metodos_teste.items():
        print(f"\n🔄 Testando {nome_metodo}...")
        
        try:
            inicio = time.time()
            
            # Determinar se deve usar GPU baseado no método
            if "_cpu" in codigo_metodo:
                use_gpu = False
            elif "_gpu" in codigo_metodo or "_cuda" in codigo_metodo:
                use_gpu = True
            else:
                use_gpu = use_gpu_default
            
            texto = OCR.extrair_texto(arquivo_path, ocr=codigo_metodo, is_save_txt=False, use_gpu=use_gpu)
            fim = time.time()
            
            tempo_execucao = fim - inicio
            caracteres = len(texto)
            chars_por_segundo = caracteres / tempo_execucao if tempo_execucao > 0 else 0
            
            resultados[nome_metodo] = {
                'tempo': tempo_execucao,
                'caracteres': caracteres,
                'chars_por_seg': chars_por_segundo,
                'use_gpu': use_gpu,
                'metodo': codigo_metodo,
                'sucesso': True
            }
            
            gpu_icon = "🚀" if use_gpu else "🖥️"
            print(f"✅ {nome_metodo}: {tempo_execucao:.2f}s | {caracteres} chars | {chars_por_segundo:.1f} chars/s {gpu_icon}")
            
        except Exception as e:
            print(f"❌ {nome_metodo}: Erro - {e}")
            resultados[nome_metodo] = {
                'tempo': None,
                'caracteres': None,
                'chars_por_seg': None,
                'use_gpu': use_gpu_default,
                'metodo': codigo_metodo,
                'sucesso': False,
                'erro': str(e)
            }
    
    # Exibir resumo comparativo
    print(f"\n=== RESUMO COMPARATIVO ===")
    metodos_sucesso = {k: v for k, v in resultados.items() if v['sucesso']}
    
    if metodos_sucesso:
        # Encontrar o mais rápido
        mais_rapido = min(metodos_sucesso.items(), key=lambda x: x[1]['tempo'])
        mais_eficiente = max(metodos_sucesso.items(), key=lambda x: x[1]['chars_por_seg'])
        
        print(f"🏆 Mais Rápido: {mais_rapido[0]} ({mais_rapido[1]['tempo']:.2f}s)")
        print(f"⚡ Mais Eficiente: {mais_eficiente[0]} ({mais_eficiente[1]['chars_por_seg']:.1f} chars/s)")
        
        # Tabela comparativa
        print(f"\n{'Método':<35} {'Tempo (s)':<12} {'Chars/s':<12} {'GPU':<8} {'Status'}")
        print("-" * 80)
        for metodo, dados in resultados.items():
            if dados['sucesso']:
                gpu_status = "🚀 Sim" if dados.get('use_gpu', False) else "🖥️  Não"
                print(f"{metodo:<35} {dados['tempo']:<12.2f} {dados['chars_por_seg']:<12.1f} {gpu_status:<8} ✅")
            else:
                print(f"{metodo:<35} {'N/A':<12} {'N/A':<12} {'N/A':<8} ❌")
    else:
        print("❌ Nenhum método OCR funcionou corretamente.")
    
    return resultados

def main():
    """Função principal."""
    print("🚀 TESTE DE PERFORMANCE OCR COM GPU")
    print("=" * 50)
    
    # Verificar status da GPU
    gpu_status = testar_gpu_status()
    
    # Executar benchmark
    print("\n" + "="*50)
    print("ESCOLHA O TIPO DE TESTE:")
    print("1. Teste completo (GPU + CPU para cada método)")
    print("2. Teste rápido (apenas configuração padrão)")
    
    try:
        escolha = input("\nDigite sua escolha (1 ou 2) [padrão: 1]: ").strip()
        test_both = escolha != "2"
    except (KeyboardInterrupt, EOFError):
        print("\nUsando configuração padrão...")
        test_both = True
    
    resultados = benchmark_ocr_methods(test_both_gpu_cpu=test_both)
    
    # Recomendações
    print(f"\n=== RECOMENDAÇÕES ===")
    
    if gpu_status['pytorch_cuda'] and OCR.EASYOCR_AVAILABLE:
        print("✅ Recomendado: Use EasyOCR com GPU para melhor performance")
    elif gpu_status['paddlepaddle_gpu'] and OCR.PADDLEOCR_AVAILABLE:
        print("✅ Recomendado: Use PaddleOCR com GPU para melhor performance")
    elif OCR.PADDLEOCR_AVAILABLE:
        print("⚠️  Recomendado: Use PaddleOCR (CPU) - geralmente mais rápido que PyTesseract")
    else:
        print("💡 Recomendado: Use PyTesseract Paralelo para melhor performance em CPU")
    
    print("\n📝 Nota: PyTesseract não suporta GPU nativamente.")
    print("   Para aceleração GPU, use EasyOCR ou PaddleOCR.")

if __name__ == "__main__":
    main()
