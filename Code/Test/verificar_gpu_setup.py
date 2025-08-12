#!/usr/bin/env python3
"""
Script para verificar e configurar dependências GPU para OCR.
"""

import subprocess
import sys
import os

def executar_comando(comando):
    """Executa um comando shell e retorna o resultado."""
    try:
        resultado = subprocess.run(comando, shell=True, capture_output=True, text=True, check=True)
        return resultado.stdout.strip(), True
    except subprocess.CalledProcessError as e:
        return e.stderr.strip(), False

def verificar_nvidia_gpu():
    """Verifica se há GPU NVIDIA disponível."""
    print("=== VERIFICAÇÃO DE GPU NVIDIA ===")
    
    # Verificar nvidia-smi
    output, sucesso = executar_comando("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits")
    
    if sucesso:
        print("✅ GPU NVIDIA detectada:")
        linhas = output.split('\n')
        for i, linha in enumerate(linhas):
            if linha.strip():
                nome, memoria = linha.split(', ')
                print(f"   GPU {i}: {nome.strip()} ({memoria.strip()} MB)")
        return True
    else:
        print("❌ Nenhuma GPU NVIDIA detectada ou nvidia-smi não instalado")
        return False

def verificar_cuda():
    """Verifica instalação do CUDA."""
    print("\n=== VERIFICAÇÃO DO CUDA ===")
    
    # Verificar nvcc
    output, sucesso = executar_comando("nvcc --version")
    
    if sucesso:
        # Extrair versão CUDA
        linhas = output.split('\n')
        for linha in linhas:
            if 'release' in linha.lower():
                print(f"✅ CUDA instalado: {linha.strip()}")
                return True
    
    print("❌ CUDA não encontrado ou nvcc não está no PATH")
    return False

def verificar_pytorch_cuda():
    """Verifica se PyTorch tem suporte CUDA."""
    print("\n=== VERIFICAÇÃO PYTORCH + CUDA ===")
    
    try:
        import torch
        cuda_disponivel = torch.cuda.is_available()
        
        if cuda_disponivel:
            print(f"✅ PyTorch com CUDA disponível")
            print(f"   Versão CUDA: {torch.version.cuda}")
            print(f"   Número de GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("❌ PyTorch instalado mas sem suporte CUDA")
            return False
            
    except ImportError:
        print("❌ PyTorch não instalado")
        return False

def verificar_paddlepaddle_gpu():
    """Verifica se PaddlePaddle tem suporte GPU."""
    print("\n=== VERIFICAÇÃO PADDLEPADDLE + GPU ===")
    
    try:
        import paddle
        gpu_disponivel = paddle.is_compiled_with_cuda()
        
        if gpu_disponivel:
            print("✅ PaddlePaddle com GPU disponível")
            return True
        else:
            print("❌ PaddlePaddle instalado mas sem suporte GPU")
            return False
            
    except ImportError:
        print("❌ PaddlePaddle não instalado")
        return False

def instalar_dependencias_gpu():
    """Sugere comandos para instalar dependências GPU."""
    print("\n=== INSTALAÇÃO DE DEPENDÊNCIAS GPU ===")
    
    print("Para instalar suporte GPU, execute os comandos abaixo:")
    print()
    
    # PyTorch com CUDA
    print("📦 PyTorch com CUDA (para EasyOCR):")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print()
    
    # PaddlePaddle GPU
    print("📦 PaddlePaddle GPU (para PaddleOCR):")
    print("   pip install paddlepaddle-gpu")
    print()
    
    # EasyOCR
    print("📦 EasyOCR:")
    print("   pip install easyocr")
    print()
    
    print("⚠️  IMPORTANTE:")
    print("   - Certifique-se de ter drivers NVIDIA atualizados")
    print("   - CUDA Toolkit deve estar instalado (versão compatível)")
    print("   - Reinicie o terminal após a instalação")

def testar_performance_basica():
    """Testa performance básica de GPU vs CPU."""
    print("\n=== TESTE DE PERFORMANCE ===")
    
    try:
        import time
        import torch
        
        if torch.cuda.is_available():
            # Teste simples de performance GPU vs CPU
            size = 1000
            
            # CPU
            start = time.time()
            cpu_tensor = torch.randn(size, size)
            cpu_result = torch.mm(cpu_tensor, cpu_tensor)
            cpu_time = time.time() - start
            
            # GPU
            start = time.time()
            gpu_tensor = torch.randn(size, size).cuda()
            gpu_result = torch.mm(gpu_tensor, gpu_tensor)
            torch.cuda.synchronize()  # Garantir que a operação termine
            gpu_time = time.time() - start
            
            print(f"⏱️  Multiplicação de matrizes {size}x{size}:")
            print(f"   CPU: {cpu_time:.4f}s")
            print(f"   GPU: {gpu_time:.4f}s")
            print(f"   Speedup: {cpu_time/gpu_time:.2f}x")
            
        else:
            print("❌ GPU não disponível para teste")
            
    except ImportError:
        print("❌ PyTorch não instalado - não é possível testar")

def main():
    """Função principal."""
    print("🔧 CONFIGURAÇÃO GPU PARA OCR")
    print("=" * 50)
    
    # Verificações
    gpu_nvidia = verificar_nvidia_gpu()
    cuda_instalado = verificar_cuda()
    pytorch_cuda = verificar_pytorch_cuda()
    paddle_gpu = verificar_paddlepaddle_gpu()
    
    # Resumo
    print("\n=== RESUMO ===")
    print(f"GPU NVIDIA: {'✅' if gpu_nvidia else '❌'}")
    print(f"CUDA: {'✅' if cuda_instalado else '❌'}")
    print(f"PyTorch + CUDA: {'✅' if pytorch_cuda else '❌'}")
    print(f"PaddlePaddle + GPU: {'✅' if paddle_gpu else '❌'}")
    
    # Recomendações
    if gpu_nvidia and cuda_instalado:
        if not pytorch_cuda and not paddle_gpu:
            print("\n💡 RECOMENDAÇÃO: Instale bibliotecas com suporte GPU")
            instalar_dependencias_gpu()
        elif pytorch_cuda or paddle_gpu:
            print("\n🎉 CONFIGURAÇÃO OK: Você pode usar aceleração GPU!")
            testar_performance_basica()
    else:
        print("\n⚠️  LIMITAÇÃO: Sem GPU NVIDIA ou CUDA - apenas CPU disponível")
        print("   PyTesseract funcionará normalmente em CPU")
        print("   EasyOCR e PaddleOCR também funcionam em CPU (mais lentos)")

if __name__ == "__main__":
    main()
