#!/usr/bin/env python3
"""
Sistema de Testes Sistemático para Algoritmos OCR
Implementa avaliação com N iterações para comparar performance de diferentes métodos OCR.
"""

import sys
import os
import time
import csv
import json
import statistics
from pathlib import Path
from pdf2image import convert_from_path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importar bibliotecas para métricas de qualidade
try:
    import Levenshtein
    LEVENSHTEIN_AVAILABLE = True
except ImportError:
    print("⚠️  Levenshtein não disponível. Execute: pip install python-Levenshtein")
    LEVENSHTEIN_AVAILABLE = False

try:
    from jiwer import wer, cer
    JIWER_AVAILABLE = True
except ImportError:
    print("⚠️  jiwer não disponível. Execute: pip install jiwer")
    JIWER_AVAILABLE = False

# Adicionar path do projeto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from Code import OCR_Extrator_PDF_Matricula

@dataclass
class ResultadoIteracao:
    """Classe para armazenar resultado de uma iteração OCR."""
    documento: str
    metodo_ocr: str
    iteracao: int
    tempo_segundos: float
    num_caracteres: int
    num_paginas: int
    similaridade_levenshtein: float = 0.0
    word_error_rate: float = 0.0
    char_error_rate: float = 0.0
    erro: Optional[str] = None
    taxa_caracteres_por_segundo: float = 0.0
    taxa_paginas_por_segundo: float = 0.0
    
    def __post_init__(self):
        if self.tempo_segundos > 0:
            self.taxa_caracteres_por_segundo = self.num_caracteres / self.tempo_segundos
            self.taxa_paginas_por_segundo = self.num_paginas / self.tempo_segundos

@dataclass
class EstatisticasMetodo:
    """Estatísticas de performance de um método OCR."""
    metodo: str
    total_documentos: int
    sucessos: int
    falhas: int
    taxa_sucesso: float
    tempo_medio: float
    tempo_mediano: float
    tempo_desvio: float
    caracteres_por_segundo_medio: float
    paginas_por_segundo_medio: float
    similaridade_levenshtein_media: float
    word_error_rate_medio: float
    char_error_rate_medio: float
    
class TestadorOCRSistematico:
    """Classe principal para testes sistemáticos de OCR."""
    
    def __init__(self, n_iteracoes: int = 5, random_seed: int = 42):
        self.n_iteracoes = n_iteracoes
        self.random_seed = random_seed
        self.resultados: List[ResultadoIteracao] = []
        self.documentos_disponiveis = []
        self.metodos_ocr = []
        
        # Configurar diretórios
        self.base_dir = Path(__file__).parent
        self.pdf_dir = self.base_dir / "Arquivos" / "PDF"
        self.truth_dir = self.base_dir / "Arquivos" / "truth_textos"
        self.resultados_dir = self.base_dir / "resultados_testes"
        self.resultados_dir.mkdir(exist_ok=True)
        
        # Verificar dependências
        self._verificar_dependencias()
        
    def _verificar_dependencias(self):
        """Verifica se as bibliotecas necessárias estão disponíveis."""
        print(f"\n🔍 VERIFICANDO DEPENDÊNCIAS:")
        print(f"   📚 Levenshtein: {'✅ Disponível' if LEVENSHTEIN_AVAILABLE else '❌ Não disponível'}")
        print(f"   📚 jiwer (WER/CER): {'✅ Disponível' if JIWER_AVAILABLE else '❌ Não disponível'}")
        
        if not LEVENSHTEIN_AVAILABLE:
            print(f"   💡 Para instalar: pip install python-Levenshtein")
        if not JIWER_AVAILABLE:
            print(f"   💡 Para instalar: pip install jiwer")
        print()
        
    def carregar_documentos(self, documentos: Optional[List[str]] = None):
        """Carrega lista de documentos para teste."""
        if documentos:
            # Limpar extensões tanto .txt quanto .pdf
            self.documentos_disponiveis = []
            for doc in documentos:
                doc_limpo = doc.removesuffix(".txt").removesuffix(".pdf")
                self.documentos_disponiveis.append(doc_limpo)
            print(f"📄 Documentos processados: {self.documentos_disponiveis}")
        else:
            # Buscar todos os PDFs disponíveis
            self.documentos_disponiveis = [
                pdf.stem for pdf in self.pdf_dir.glob("*.pdf")
            ]
        
        print(f"📄 {len(self.documentos_disponiveis)} documentos carregados para teste")
        
    def configurar_metodos_ocr(self, metodos: Optional[List[str]] = None):
        """Configura métodos OCR para teste."""
        if metodos:
            self.metodos_ocr = metodos
        else:
            # Métodos padrão disponíveis
            self.metodos_ocr = [
                OCR_Extrator_PDF_Matricula.OCR_PYTESSERACT,
                OCR_Extrator_PDF_Matricula.OCR_PYTESSERACT_PARALLEL,
                OCR_Extrator_PDF_Matricula.OCR_PADDLE_CPU,
                OCR_Extrator_PDF_Matricula.OCR_PADDLE_GPU,
                # OCR_Extrator_PDF_Matricula.OCR_EASYOCR_CPU,  # Descomentar quando PyTorch estiver OK
                # OCR_Extrator_PDF_Matricula.OCR_EASYOCR_GPU,
            ]
        
        print(f"🔧 {len(self.metodos_ocr)} métodos OCR configurados: {self.metodos_ocr}")
    
    def carregar_texto_ground_truth(self, documento: str) -> Optional[str]:
        """Carrega texto ground truth de um documento."""
        truth_path = self.truth_dir / f"{documento}.txt"
        print(f"🔍 Procurando ground truth: {truth_path}")
        
        if truth_path.exists():
            try:
                with open(truth_path, 'r', encoding='utf-8') as f:
                    conteudo = f.read()
                    print(f"✅ Ground truth carregado: {len(conteudo)} caracteres")
                    return conteudo
            except Exception as e:
                print(f"⚠️  Erro ao carregar ground truth de {documento}: {e}")
        else:
            print(f"❌ Arquivo ground truth não encontrado: {truth_path}")
            print(f"📁 Diretório truth_textos: {self.truth_dir}")
            # Listar arquivos disponíveis para debug
            if self.truth_dir.exists():
                arquivos_disponiveis = list(self.truth_dir.glob("*.txt"))
                print(f"📄 Arquivos disponíveis: {[f.name for f in arquivos_disponiveis[:5]]}...")
            
        return None
    
    def calcular_metricas_qualidade(self, texto_ocr: str, texto_ground_truth: str) -> tuple:
        """
        Calcula métricas de qualidade comparando texto OCR com ground truth.
        
        Returns:
            tuple: (similaridade_levenshtein, word_error_rate, char_error_rate)
        """
        if not texto_ground_truth:
            print(f"⚠️  Ground truth não encontrado - usando valores padrão")
            return 0.0, 1.0, 1.0
        
        # Similaridade Levenshtein
        similaridade_lev = 0.0
        if LEVENSHTEIN_AVAILABLE:
            try:
                similaridade_lev = Levenshtein.ratio(texto_ocr, texto_ground_truth)
            except Exception as e:
                print(f"   ⚠️  Erro ao calcular Levenshtein: {e}")
        else:
            print(f"   ⚠️  Biblioteca Levenshtein não disponível")
        
        # Word Error Rate e Character Error Rate
        wer_score = 1.0
        cer_score = 1.0
        if JIWER_AVAILABLE:
            try:
                wer_score = wer(texto_ground_truth, texto_ocr)
                cer_score = cer(texto_ground_truth, texto_ocr)
            except Exception as e:
                print(f"   ⚠️  Erro ao calcular WER/CER: {e}")
        else:
            print(f"   ⚠️  Biblioteca jiwer não disponível")
        
        return similaridade_lev, wer_score, cer_score
        
    def dividir_documentos_k_fold(self) -> List[List[str]]:
        """REMOVIDO: Não usar K-Fold."""
        pass
        
    def contar_paginas_pdf(self, pdf_path: Path) -> int:
        """Conta número de páginas de um PDF."""
        try:
            paginas = len(convert_from_path(pdf_path, dpi=72))
            return paginas
        except Exception as e:
            print(f"⚠️  Erro ao contar páginas de {pdf_path.name}: {e}")
            return 1  # Assumir 1 página se der erro
            
    def executar_teste_iteracao(self, documento: str, metodo: str, iteracao: int) -> ResultadoIteracao:
        """Executa teste OCR em uma iteração específica."""
        pdf_path = self.pdf_dir / f"{documento}.pdf"
        
        if not pdf_path.exists():
            return ResultadoIteracao(
                documento=documento,
                metodo_ocr=metodo,
                iteracao=iteracao,
                tempo_segundos=0,
                num_caracteres=0,
                num_paginas=0,
                similaridade_levenshtein=0.0,
                word_error_rate=1.0,
                char_error_rate=1.0,
                erro=f"Arquivo {pdf_path.name} não encontrado"
            )
        
        num_paginas = self.contar_paginas_pdf(pdf_path)
        
        try:
            print(f"   🔄 Iteração {iteracao}: {documento}.pdf com {metodo}...")
            inicio = time.time()
            texto_ocr = OCR_Extrator_PDF_Matricula.extrair_texto(
                pdf_path, 
                ocr=metodo, 
                is_save_txt=False  # Não salvar TXT durante testes
            )
            fim = time.time()
            
            duracao = fim - inicio
            num_caracteres = len(texto_ocr.strip())
            
            # Carregar ground truth e calcular métricas de qualidade
            texto_ground_truth = self.carregar_texto_ground_truth(documento)
            similaridade_lev, wer_score, cer_score = self.calcular_metricas_qualidade(
                texto_ocr, texto_ground_truth
            )
            
            return ResultadoIteracao(
                documento=documento,
                metodo_ocr=metodo,
                iteracao=iteracao,
                tempo_segundos=round(duracao, 3),
                num_caracteres=num_caracteres,
                num_paginas=num_paginas,
                similaridade_levenshtein=similaridade_lev,
                word_error_rate=wer_score,
                char_error_rate=cer_score
            )
            
        except Exception as e:
            return ResultadoIteracao(
                documento=documento,
                metodo_ocr=metodo,
                iteracao=iteracao,
                tempo_segundos=0,
                num_caracteres=0,
                num_paginas=num_paginas,
                similaridade_levenshtein=0.0,
                word_error_rate=1.0,
                char_error_rate=1.0,
                erro=str(e)
            )
    
    def executar_testes_completos(self):
        """Executa todos os testes com N iterações."""
        print(f"\n🚀 INICIANDO TESTES SISTEMÁTICOS OCR")
        print(f"� Iterações por método: {self.n_iteracoes}")
        print(f"📄 Documentos: {len(self.documentos_disponiveis)}")
        print(f"🔧 Métodos: {len(self.metodos_ocr)}")
        print("=" * 80)
        
        total_testes = len(self.documentos_disponiveis) * len(self.metodos_ocr) * self.n_iteracoes
        teste_atual = 0
        
        for metodo in self.metodos_ocr:
            print(f"\n� MÉTODO: {metodo}")
            
            for documento in self.documentos_disponiveis:
                print(f"\n� Documento: {documento}.pdf")
                
                for iteracao in range(1, self.n_iteracoes + 1):
                    teste_atual += 1
                    progresso = (teste_atual / total_testes) * 100
                    print(f"📈 Progresso: {progresso:.1f}% ({teste_atual}/{total_testes})")
                    
                    resultado = self.executar_teste_iteracao(documento, metodo, iteracao)
                    self.resultados.append(resultado)
                    
                    if resultado.erro:
                        print(f"   ❌ Iteração {iteracao}: {resultado.erro}")
                    else:
                        print(f"   ✅ Iteração {iteracao}: {resultado.tempo_segundos}s, "
                              f"{resultado.num_caracteres} chars, "
                              f"Sim: {resultado.similaridade_levenshtein:.3f}")
        
        print(f"\n🎉 TESTES CONCLUÍDOS! {len(self.resultados)} resultados obtidos")
        
    def calcular_estatisticas(self) -> Dict[str, EstatisticasMetodo]:
        """Calcula estatísticas por método OCR."""
        estatisticas = {}
        
        for metodo in self.metodos_ocr:
            resultados_metodo = [r for r in self.resultados if r.metodo_ocr == metodo]
            sucessos = [r for r in resultados_metodo if r.erro is None]
            falhas = [r for r in resultados_metodo if r.erro is not None]
            
            if sucessos:
                # Métricas de tempo
                tempos = [r.tempo_segundos for r in sucessos]
                chars_por_seg = [r.taxa_caracteres_por_segundo for r in sucessos]
                pags_por_seg = [r.taxa_paginas_por_segundo for r in sucessos]
                
                # Métricas de qualidade - filtrar valores válidos
                similaridades = [r.similaridade_levenshtein for r in sucessos if r.similaridade_levenshtein is not None]
                wer_scores = [r.word_error_rate for r in sucessos if r.word_error_rate is not None]
                cer_scores = [r.char_error_rate for r in sucessos if r.char_error_rate is not None]
                
                # Calcular médias das métricas de qualidade
                similaridade_media = statistics.mean(similaridades) if similaridades else 0.0
                wer_medio = statistics.mean(wer_scores) if wer_scores else 1.0
                cer_medio = statistics.mean(cer_scores) if cer_scores else 1.0
                
                estatisticas[metodo] = EstatisticasMetodo(
                    metodo=metodo,
                    total_documentos=len(set(r.documento for r in resultados_metodo)),
                    sucessos=len(sucessos),
                    falhas=len(falhas),
                    taxa_sucesso=len(sucessos) / len(resultados_metodo),
                    tempo_medio=statistics.mean(tempos),
                    tempo_mediano=statistics.median(tempos),
                    tempo_desvio=statistics.stdev(tempos) if len(tempos) > 1 else 0,
                    caracteres_por_segundo_medio=statistics.mean(chars_por_seg),
                    paginas_por_segundo_medio=statistics.mean(pags_por_seg),
                    similaridade_levenshtein_media=similaridade_media,
                    word_error_rate_medio=wer_medio,
                    char_error_rate_medio=cer_medio
                )
            else:
                estatisticas[metodo] = EstatisticasMetodo(
                    metodo=metodo,
                    total_documentos=len(set(r.documento for r in resultados_metodo)),
                    sucessos=0,
                    falhas=len(falhas),
                    taxa_sucesso=0.0,
                    tempo_medio=0.0,
                    tempo_mediano=0.0,
                    tempo_desvio=0.0,
                    caracteres_por_segundo_medio=0.0,
                    paginas_por_segundo_medio=0.0,
                    similaridade_levenshtein_media=0.0,
                    word_error_rate_medio=1.0,
                    char_error_rate_medio=1.0
                )
        
        return estatisticas
        
    def salvar_resultados(self):
        """Salva resultados em diferentes formatos."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 1. CSV detalhado
        csv_path = self.resultados_dir / f"resultados_detalhados_{timestamp}.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'documento', 'metodo_ocr', 'iteracao', 'tempo_segundos', 
                'num_caracteres', 'num_paginas', 'erro',
                'taxa_caracteres_por_segundo', 'taxa_paginas_por_segundo',
                'similaridade_levenshtein', 'word_error_rate', 'char_error_rate'
            ])
            
            for resultado in self.resultados:
                writer.writerow([
                    resultado.documento,
                    resultado.metodo_ocr,
                    resultado.iteracao,
                    resultado.tempo_segundos,
                    resultado.num_caracteres,
                    resultado.num_paginas,
                    resultado.erro or '',
                    resultado.taxa_caracteres_por_segundo,
                    resultado.taxa_paginas_por_segundo,
                    resultado.similaridade_levenshtein,
                    resultado.word_error_rate,
                    resultado.char_error_rate
                ])
        
        # 2. JSON detalhado
        json_path = self.resultados_dir / f"resultados_detalhados_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump([asdict(r) for r in self.resultados], f, indent=2, ensure_ascii=False)
        
        # 3. CSV de estatísticas (formato solicitado)
        estatisticas = self.calcular_estatisticas()
        stats_csv_path = self.resultados_dir / f"estatisticas_{timestamp}.csv"
        with open(stats_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'metodo', 'total_documentos', 'sucessos', 'falhas', 'taxa_sucesso',
                'tempo_medio', 'tempo_mediano', 'tempo_desvio',
                'caracteres_por_segundo_medio', 'paginas_por_segundo_medio',
                'similaridade_levenshtein', 'word_error_rate', 'char_error_rate'
            ])
            
            for stats in estatisticas.values():
                writer.writerow([
                    stats.metodo, 
                    stats.total_documentos, 
                    stats.sucessos, 
                    stats.falhas,
                    f"{stats.taxa_sucesso:.4f}",
                    f"{stats.tempo_medio:.3f}",
                    f"{stats.tempo_mediano:.3f}", 
                    f"{stats.tempo_desvio:.3f}",
                    f"{stats.caracteres_por_segundo_medio:.1f}",
                    f"{stats.paginas_por_segundo_medio:.3f}",
                    f"{stats.similaridade_levenshtein_media:.4f}",
                    f"{stats.word_error_rate_medio:.4f}",
                    f"{stats.char_error_rate_medio:.4f}"
                ])
        
        print(f"💾 Resultados salvos:")
        print(f"   📄 CSV detalhado: {csv_path}")
        print(f"   📄 JSON detalhado: {json_path}")
        print(f"   📊 Estatísticas: {stats_csv_path}")
        
        return csv_path, json_path, stats_csv_path
        
    def gerar_relatorio_console(self):
        """Gera relatório no console."""
        estatisticas = self.calcular_estatisticas()
        
        print(f"\n{'='*80}")
        print(f"📊 RELATÓRIO DE PERFORMANCE OCR")
        print(f"{'='*80}")
        
        print(f"\n📈 RANKING POR VELOCIDADE (Caracteres/segundo):")
        ranking_velocidade = sorted(
            estatisticas.values(), 
            key=lambda x: x.caracteres_por_segundo_medio, 
            reverse=True
        )
        
        for i, stats in enumerate(ranking_velocidade, 1):
            print(f"   {i}. {stats.metodo:20} | "
                  f"{stats.caracteres_por_segundo_medio:8.1f} chars/s | "
                  f"{stats.tempo_medio:6.2f}s médio | "
                  f"Taxa sucesso: {stats.taxa_sucesso:.1%}")
        
        print(f"\n🎯 RANKING POR CONFIABILIDADE (Taxa de sucesso):")
        ranking_confiabilidade = sorted(
            estatisticas.values(), 
            key=lambda x: x.taxa_sucesso, 
            reverse=True
        )
        
        for i, stats in enumerate(ranking_confiabilidade, 1):
            print(f"   {i}. {stats.metodo:20} | "
                  f"Sucesso: {stats.taxa_sucesso:6.1%} | "
                  f"Sucessos: {stats.sucessos:3d}/{stats.total_documentos} | "
                  f"Falhas: {stats.falhas}")
        
        print(f"\n🎯 RANKING POR QUALIDADE (Similaridade Levenshtein):")
        ranking_qualidade = sorted(
            estatisticas.values(), 
            key=lambda x: x.similaridade_levenshtein_media, 
            reverse=True
        )
        
        for i, stats in enumerate(ranking_qualidade, 1):
            print(f"   {i}. {stats.metodo:20} | "
                  f"Similaridade: {stats.similaridade_levenshtein_media:6.3f} | "
                  f"WER: {stats.word_error_rate_medio:6.3f} | "
                  f"CER: {stats.char_error_rate_medio:6.3f}")
        
        print(f"\n⚡ RESUMO GERAL:")
        total_testes = len(self.resultados)
        total_sucessos = len([r for r in self.resultados if r.erro is None])
        total_falhas = len([r for r in self.resultados if r.erro is not None])
        
        print(f"   📊 Total de testes: {total_testes}")
        print(f"   ✅ Sucessos: {total_sucessos} ({total_sucessos/total_testes:.1%})")
        print(f"   ❌ Falhas: {total_falhas} ({total_falhas/total_testes:.1%})")
        print(f"   🔧 Métodos testados: {len(self.metodos_ocr)}")
        print(f"   📄 Documentos testados: {len(self.documentos_disponiveis)}")
        print(f"   📁 Iterações por documento: {self.n_iteracoes}")

def main():
    """Função principal do script."""
    # Configuração do teste
    testador = TestadorOCRSistematico(n_iteracoes=3)  # 3 iterações por documento para teste
    
    # Documentos para teste (usar lista menor para desenvolvimento)
    documentos_teste = [
    "MATRICULAGREGO.txt", "2301.txt", "2720.txt", "4724.txt", "5494.txt", "5892.txt",
    "6971.txt", "7131.txt", "7430.txt", "12665.txt", "12688.txt",
    "12690.txt", "12867.txt", "12878.txt", "13123.txt", "13164.txt",
    "17071.txt", "17074.txt", "17433.txt", "17434.txt", "22793.txt",
    "62406.txt", "62451.txt", "62489.txt", "62497.txt", "65716.txt",
    "65717.txt", "65718.txt", "65771.txt", "65772.txt"
    ]
    
    # Métodos OCR para teste
    metodos_teste = [
     #OCR_Extrator_PDF_Matricula.OCR_PYTESSERACT,
     OCR_Extrator_PDF_Matricula.OCR_PYTESSERACT_PARALLEL,
     OCR_Extrator_PDF_Matricula.OCR_EASYOCR_GPU,  #
     #OCR_Extrator_PDF_Matricula.OCR_EASYOCR_CPU, 
     #OCR_Extrator_PDF_Matricula.OCR_GOOGLE_VISION,  # Requer credenciais
     #OCR_Extrator_PDF_Matricula.OCR_GOOGLE_GEMINI,  # Requer credenciais
     #OCR_Extrator_PDF_Matricula.OCR_AWS_TEXTRACT, # Requer credenciais
     #OCR_Extrator_PDF_Matricula.OCR_PADDLE_CPU,
     OCR_Extrator_PDF_Matricula.OCR_PADDLE_GPU,  # Requer PaddleOCR instalado
    ]
    
    # Executar testes
    testador.carregar_documentos(documentos_teste)
    testador.configurar_metodos_ocr(metodos_teste)
    testador.executar_testes_completos()
    
    # Gerar relatórios
    testador.gerar_relatorio_console()
    testador.salvar_resultados()

if __name__ == "__main__":
    main()