#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Normalizador de Texto para Testes de Similaridade
"""

import os
import re
import json
import unicodedata
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mapeamento de caracteres especiais para português
CHARS_MAP = str.maketrans({
    'ï': 'i', 'ù': 'u', 'ö': 'o', 'î': 'i', 'ñ': 'n',
    'ë': 'e', 'ì': 'i', 'ò': 'o', 'ů': 'u', 'ẽ': 'e',
    'ü': 'u', 'è': 'e', 'æ': 'a', 'å': 'a', 'ø': 'o',
    'þ': 't', 'ð': 'd', 'ß': 's', 'ł': 'l', 'đ': 'd',
    'ć': 'c', 'č': 'c', 'š': 's', 'ž': 'z', 'ý': 'y'
})

def apply_char_mapping(text: str) -> str:
    """
    Aplica mapeamento de caracteres especiais usando str.translate (mais eficiente)
    
    Args:
        text: Texto para aplicar mapeamento
        
    Returns:
        Texto com caracteres mapeados
    """
    return text.translate(CHARS_MAP)


def number_to_words_pt(num: int) -> str:
    """
    Converte número para extenso em português
    Suporta números de 0 até 999.999.999
    
    Args:
        num: Número inteiro para converter
        
    Returns:
        Número por extenso em português
    """
    if num == 0:
        return "zero"
    
    # Unidades
    ones = ["", "um", "dois", "três", "quatro", "cinco", "seis", "sete", "oito", "nove",
            "dez", "onze", "doze", "treze", "quatorze", "quinze", "dezesseis", 
            "dezessete", "dezoito", "dezenove"]
    
    # Dezenas
    tens = ["", "", "vinte", "trinta", "quarenta", "cinquenta", "sessenta", 
            "setenta", "oitenta", "noventa"]
    
    # Centenas
    hundreds = ["", "cento", "duzentos", "trezentos", "quatrocentos", "quinhentos",
                "seiscentos", "setecentos", "oitocentos", "novecentos"]
    
    if num < 0:
        return "menos " + number_to_words_pt(-num)
    
    if num < 20:
        return ones[num]
    
    if num < 100:
        if num % 10 == 0:
            return tens[num // 10]
        else:
            return tens[num // 10] + " e " + ones[num % 10]
    
    if num == 100:
        return "cem"
    
    if num < 1000:
        if num % 100 == 0:
            return hundreds[num // 100]
        else:
            return hundreds[num // 100] + " e " + number_to_words_pt(num % 100)
    
    if num < 1000000:
        thousands = num // 1000
        remainder = num % 1000
        
        if thousands == 1:
            result = "mil"
        else:
            result = number_to_words_pt(thousands) + " mil"
        
        if remainder > 0:
            # Usa "e" apenas se o resto for menor que 100
            if remainder < 100:
                result += " e " + number_to_words_pt(remainder)
            else:
                result += " " + number_to_words_pt(remainder)
        
        return result
    
    if num < 1000000000:
        millions = num // 1000000
        remainder = num % 1000000
        
        if millions == 1:
            result = "um milhão"
        else:
            result = number_to_words_pt(millions) + " milhões"
        
        if remainder > 0:
            if remainder < 100:
                result += " e " + number_to_words_pt(remainder)
            else:
                result += " " + number_to_words_pt(remainder)
        
        return result
    
    # Para números maiores que 999.999.999, retorna o número original
    return str(num)


def ordinal_to_words_pt(num: int, gender: str = 'm') -> str:
    """
    Converte número ordinal para extenso em português
    
    Args:
        num: Número ordinal
        gender: Gênero ('m' para masculino, 'f' para feminino)
        
    Returns:
        Ordinal por extenso
    """
    # Ordinais básicos masculinos
    ordinals_m = {
        1: "primeiro", 2: "segundo", 3: "terceiro", 4: "quarto", 5: "quinto",
        6: "sexto", 7: "sétimo", 8: "oitavo", 9: "nono", 10: "décimo",
        11: "décimo primeiro", 12: "décimo segundo", 13: "décimo terceiro",
        14: "décimo quarto", 15: "décimo quinto", 16: "décimo sexto",
        17: "décimo sétimo", 18: "décimo oitavo", 19: "décimo nono",
        20: "vigésimo", 21: "vigésimo primeiro", 30: "trigésimo",
        40: "quadragésimo", 50: "quinquagésimo", 60: "sexagésimo",
        70: "septuagésimo", 80: "octogésimo", 90: "nonagésimo",
        100: "centésimo"
    }
    
    # Ordinais básicos femininos
    ordinals_f = {
        1: "primeira", 2: "segunda", 3: "terceira", 4: "quarta", 5: "quinta",
        6: "sexta", 7: "sétima", 8: "oitava", 9: "nona", 10: "décima",
        11: "décima primeira", 12: "décima segunda", 13: "décima terceira",
        14: "décima quarta", 15: "décima quinta", 16: "décima sexta",
        17: "décima sétima", 18: "décima oitava", 19: "décima nona",
        20: "vigésima", 21: "vigésima primeira", 30: "trigésima",
        40: "quadragésima", 50: "quinquagésima", 60: "sexagésima",
        70: "septuagésima", 80: "octogésima", 90: "nonagésima",
        100: "centésima"
    }
    
    ordinals = ordinals_f if gender == 'f' else ordinals_m
    
    if num in ordinals:
        return ordinals[num]
    
    # Para números não mapeados, usa o cardinal
    return number_to_words_pt(num)


def advanced_number_to_text(text: str) -> str:
    """
    Conversão avançada de números e símbolos para texto
    
    Args:
        text: Texto com números e símbolos
        
    Returns:
        Texto com números convertidos para extenso
    """
    result = text
    
    # Primeiro, trata ordinais (1º, 2ª, 15º, etc.)
    def replace_ordinal(match):
        num = int(match.group(1))
        suffix = match.group(2)
        gender = 'f' if suffix in ['ª', 'a'] else 'm'
        return ordinal_to_words_pt(num, gender)
    
    # Regex para ordinais: 1º, 2ª, 15º, etc.
    result = re.sub(r'(\d+)([ºªº°])', replace_ordinal, result)
    
    # Trata números decimais (ex: 20,50 ou 1.5)
    def replace_decimal(match):
        integer_part = match.group(1)
        separator = match.group(2)
        decimal_part = match.group(3)
        
        # Converte parte inteira
        integer_text = number_to_words_pt(int(integer_part))
        
        # Converte separador
        sep_text = "vírgula" if separator == "," else "ponto"
        
        # Converte parte decimal dígito por dígito
        decimal_text = " ".join([number_to_words_pt(int(d)) for d in decimal_part])
        
        return f"{integer_text} {sep_text} {decimal_text}"
    
    # Regex para números decimais (ex: 20,50 ou 1.25)
    result = re.sub(r'(\d+)([,.](\d+))', replace_decimal, result)
    
    # Trata números inteiros restantes
    def replace_integer(match):
        num = int(match.group(0))
        return number_to_words_pt(num)
    
    # Regex para números inteiros que sobraram
    result = re.sub(r'\b\d+\b', replace_integer, result)
    
    # Trata símbolos monetários e unidades
    symbol_replacements = {
        r'R\$\s*': 'reais ',
        r'US\$\s*': 'dólares ',
        r'\$\s*': 'dólares ',
        r'€\s*': 'euros ',
        r'%': ' por cento',
        r'°C': ' graus celsius',
        r'°F': ' graus fahrenheit',
        r'km/h': ' quilômetros por hora',
        r'm/s': ' metros por segundo',
        r'\bkg\b': ' quilogramas',
        r'\bg\b': ' gramas',
        r'\bkm\b': ' quilômetros',
        r'\bcm\b': ' centímetros',
        r'\bmm\b': ' milímetros'
    }
    
    for pattern, replacement in symbol_replacements.items():
        result = re.sub(pattern, replacement, result)
    
    return result


def remove_html_tags(text: str) -> str:
    """
    Remove tags HTML usando regex
    
    Args:
        text: Texto com possíveis tags HTML
        
    Returns:
        Texto sem tags HTML
    """
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def text_cleaning(text: str) -> str:
    """
    Limpeza e normalização de texto
    IMPORTANTE: Apenas padroniza formato, não corrige ortografia
    
    Args:
        text: Texto para limpar
        
    Returns:
        Texto limpo e normalizado
    """
    if not text or text.strip() == "":
        return ""
    
    # Remove quebras de linha
    text = text.replace('\n', ' ')
    
    # Remove tags HTML
    text = remove_html_tags(text)
    
    # Remove TODOS os acentos (á→a, ç→c, ã→a, etc)
    text = unicodedata.normalize('NFD', text)
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    
    # Aplica mapeamento de caracteres especiais (ö→o, ñ→n, etc)
    text = apply_char_mapping(text)
    
    # Converte para minúsculas
    text = text.lower()
    
    # Substitui ... por .
    text = re.sub(r'[.]{3,}', '.', text)
    
    # Remove parênteses e colchetes
    text = re.sub(r'[(\[\])]', '', text)
    
    # Remove pontuação (APÓS conversão de números)
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for char in punctuations:
        text = text.replace(char, ' ')
    
    # Remove espaços múltiplos
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def normalize_text(text: str) -> Optional[str]:
    """
    Normalização completa do texto
    IMPORTANTE: Apenas padroniza, não corrige conteúdo
    
    Args:
        text: Texto para normalizar
        
    Returns:
        Texto normalizado ou None se vazio
    """
    if not text or text.strip() == "":
        return None
    
    # Converte números para texto
    text = advanced_number_to_text(text)
    
    # Aplica limpeza
    normalized = text_cleaning(text)
    
    return normalized if normalized else None


def extract_file_info(filename: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extrai informações do nome do arquivo dinamicamente
    Padrão: {video_id}_..._segment_{000}_.._{modelo}.txt
    
    Args:
        filename: Nome do arquivo
        
    Returns:
        Tupla (video_id, segment_number, modelo)
    """
    # Remove extensão
    name = filename.replace('.txt', '')
    
    # Extrai video_id (primeiros caracteres antes do primeiro _)
    video_id_match = re.match(r'^([^_]+)', name)
    if not video_id_match:
        return None, None, None
    
    video_id = video_id_match.group(1)
    
    # Extrai número do segmento
    segment_match = re.search(r'segment_(\d{3,4})', name)
    if not segment_match:
        return None, None, None
    
    segment_number = segment_match.group(1)
    
    # Extrai modelo (wav2vec2 ou whisper)
    if 'wav2vec2' in name:
        modelo = 'wav2vec2'
    elif 'whisper' in name:
        modelo = 'whisper'
    else:
        return None, None, None
    
    return video_id, segment_number, modelo


def read_text_file(filepath: Path) -> Optional[str]:
    """
    Lê arquivo de texto com tratamento de encoding
    
    Args:
        filepath: Caminho do arquivo
        
    Returns:
        Conteúdo do arquivo ou None em caso de erro
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except UnicodeDecodeError:
        # Fallback para encoding latin-1
        try:
            with open(filepath, 'r', encoding='latin-1') as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Erro ao ler {filepath}: {e}")
            return None
    except Exception as e:
        logger.error(f"Erro ao ler {filepath}: {e}")
        return None


def find_stt_directories(session_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Busca automaticamente os diretórios STT-whisper e STT-wav2vec2
    
    Args:
        session_dir: Diretório da sessão
        
    Returns:
        Tupla (whisper_dir, wav2vec2_dir)
    """
    # Padrão esperado: session_dir/stt_results/stt_results/
    stt_base = session_dir / 'stt_results' / 'stt_results'
    
    if not stt_base.exists():
        # Tenta alternativa: session_dir/stt_results/
        stt_base = session_dir / 'stt_results'
    
    whisper_dir = stt_base / 'STT-whisper'
    wav2vec2_dir = stt_base / 'STT-wav2vec2'
    
    # Verifica se os diretórios existem
    whisper_exists = whisper_dir.exists() and whisper_dir.is_dir()
    wav2vec2_exists = wav2vec2_dir.exists() and wav2vec2_dir.is_dir()
    
    return (whisper_dir if whisper_exists else None,
            wav2vec2_dir if wav2vec2_exists else None)


def find_all_sessions(base_dir: str = "audios_baixados/output") -> List[Path]:
    """
    Busca todas as sessões disponíveis automaticamente
    
    Args:
        base_dir: Diretório base onde estão as sessões
        
    Returns:
        Lista de caminhos das sessões encontradas
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        logger.warning(f"Diretório base não encontrado: {base_path}")
        return []
    
    # Busca todas as pastas que contenham stt_results
    sessions = []
    
    for item in base_path.iterdir():
        if item.is_dir():
            # Verifica se tem stt_results dentro
            stt_path = item / 'stt_results'
            if stt_path.exists():
                sessions.append(item)
                logger.debug(f"Sessão encontrada: {item.name}")
    
    return sessions


def process_all_sessions(base_dir: str = "audios_baixados/output") -> Dict:
    """
    Processa todas as sessões encontradas automaticamente
    
    Args:
        base_dir: Diretório base onde estão as sessões
        
    Returns:
        Dicionário com resultados do processamento de todas as sessões
    """
    logger.info("Buscando sessões automaticamente...")
    
    sessions = find_all_sessions(base_dir)
    
    if not sessions:
        error_msg = f"Nenhuma sessão encontrada em {base_dir}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}
    
    logger.info(f"Encontradas {len(sessions)} sessões para processar")
    
    results = {
        "success": True,
        "total_sessions": len(sessions),
        "processed_sessions": [],
        "failed_sessions": []
    }
    
    for session_path in sessions:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processando sessão: {session_path.name}")
        logger.info(f"{'='*60}")
        
        result = process_stt_results(str(session_path))
        
        if result["success"]:
            results["processed_sessions"].append({
                "session_name": session_path.name,
                "session_path": str(session_path),
                "output_files": result["output_files"],
                "total_videos": result["total_videos"],
                "total_segments": result["total_segments"]
            })
        else:
            results["failed_sessions"].append({
                "session_name": session_path.name,
                "session_path": str(session_path),
                "error": result.get("error")
            })
    
    # Resumo final
    logger.info(f"\n{'='*60}")
    logger.info("RESUMO DO PROCESSAMENTO")
    logger.info(f"{'='*60}")
    logger.info(f"Total de sessões: {results['total_sessions']}")
    logger.info(f"Processadas com sucesso: {len(results['processed_sessions'])}")
    logger.info(f"Falharam: {len(results['failed_sessions'])}")
    
    return results


def process_stt_results(session_dir: str) -> Dict:
    """
    Processa resultados STT de uma sessão automaticamente
    Busca arquivos .txt nos diretórios STT-whisper e STT-wav2vec2
    
    Args:
        session_dir: Diretório da sessão (ex: "audios_baixados/output/teste_com_token")
        
    Returns:
        Dicionário com resultado do processamento
    """
    session_path = Path(session_dir)
    
    if not session_path.exists():
        error_msg = f"Diretório da sessão não encontrado: {session_path}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}
    
    logger.info(f"Processando sessão: {session_path}")
    
    # Busca diretórios STT automaticamente
    whisper_dir, wav2vec2_dir = find_stt_directories(session_path)
    
    if not whisper_dir and not wav2vec2_dir:
        error_msg = "Nenhum diretório STT encontrado"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}
    
    # Coleta todos os arquivos .txt
    txt_files = []
    
    if whisper_dir:
        whisper_files = list(whisper_dir.glob("*.txt"))
        txt_files.extend(whisper_files)
        logger.info(f"Encontrados {len(whisper_files)} arquivos Whisper")
    
    if wav2vec2_dir:
        wav2vec2_files = list(wav2vec2_dir.glob("*.txt"))
        txt_files.extend(wav2vec2_files)
        logger.info(f"Encontrados {len(wav2vec2_files)} arquivos WAV2VEC2")
    
    if not txt_files:
        error_msg = "Nenhum arquivo .txt encontrado"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}
    
    logger.info(f"Total de arquivos .txt: {len(txt_files)}")
    
    # Agrupa arquivos por video_id e segment
    grouped_files = {}
    
    for txt_file in txt_files:
        video_id, segment_number, modelo = extract_file_info(txt_file.name)
        
        if not all([video_id, segment_number, modelo]):
            logger.warning(f"Erro ao extrair informações de: {txt_file.name}")
            continue
        
        # Chave única para agrupar
        key = f"{video_id}_{segment_number}"
        
        if key not in grouped_files:
            grouped_files[key] = {'video_id': video_id, 'segment': segment_number}
        
        # Lê conteúdo do arquivo
        content = read_text_file(txt_file)
        if content:
            grouped_files[key][f"{modelo}_file"] = txt_file.name
            grouped_files[key][f"{modelo}_original"] = content
            grouped_files[key][f"{modelo}_normalized"] = normalize_text(content)
    
    if not grouped_files:
        error_msg = "Nenhum arquivo válido processado"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}
    
    # Agrupa por video_id para criar JSONs separados
    videos = {}
    for key, data in grouped_files.items():
        video_id = data['video_id']
        if video_id not in videos:
            videos[video_id] = {}
        videos[video_id][key] = data
    
    # Cria JSON para cada video_id
    output_files = []
    
    for video_id, segments in videos.items():
        normalized_pairs = {}
        valid_pairs = 0
        
        # Ordena segmentos por número (0000, 0001, 0002, etc.)
        sorted_segments = sorted(segments.items(), key=lambda x: x[0])
        
        for segment_key, data in sorted_segments:
            normalized_pairs[segment_key] = {
                'wav2vec2_original': data.get('wav2vec2_original'),
                'wav2vec2_normalized': data.get('wav2vec2_normalized'),
                'whisper_original': data.get('whisper_original'), 
                'whisper_normalized': data.get('whisper_normalized'),
                'segment_filename': f"{segment_key}.wav"
            }
            
            # Conta pares válidos (com ambos os modelos normalizados)
            if (data.get('wav2vec2_normalized') and data.get('whisper_normalized')):
                valid_pairs += 1
        
        # Cria estrutura final
        result = {
            "metadata": {
                "processing_date": datetime.now().isoformat(),
                "wav2vec2_source": "arquivos *wav2vec2.txt",
                "whisper_source": "arquivos *whisper.txt", 
                "total_pairs": len(normalized_pairs),
                "valid_pairs": valid_pairs,
                "video_id": video_id
            },
            "normalized_pairs": normalized_pairs
        }
        
        # Salva JSON no diretório stt_results
        output_dir = session_path / 'stt_results'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{video_id}_normalized_text.json"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Arquivo salvo: {output_file}")
            logger.info(f"Video ID: {video_id}")
            logger.info(f"Total de segmentos: {len(normalized_pairs)}")
            logger.info(f"Pares válidos: {valid_pairs}")
            logger.info("-" * 50)
            
            output_files.append(str(output_file))
            
        except Exception as e:
            logger.error(f"Erro ao salvar {output_file}: {e}")
            return {"success": False, "error": str(e)}
    
    return {
        "success": True,
        "output_files": output_files,
        "total_videos": len(videos),
        "total_segments": len(grouped_files)
    }


def main():
    """
    Função principal para uso standalone
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Normalizador de Texto para STT')
    parser.add_argument('session_dir', type=str, nargs='?', default=None,
                       help='Diretório da sessão específica (opcional). Se não fornecido, processa todas as sessões.')
    parser.add_argument('--base-dir', type=str, default='audios_baixados/output',
                       help='Diretório base onde estão as sessões (padrão: audios_baixados/output)')
    parser.add_argument('--all', action='store_true',
                       help='Processar todas as sessões automaticamente')
    
    args = parser.parse_args()
    
    logger.info("NORMALIZADOR DE TEXTO PARA SIMILARIDADE")
    logger.info("=" * 50)
    
    # Se --all ou nenhum session_dir fornecido, processa todas as sessões
    if args.all or args.session_dir is None:
        logger.info("Modo: Processamento automático de todas as sessões")
        result = process_all_sessions(args.base_dir)
        
        if result["success"]:
            logger.info("\nProcessamento concluído!")
            logger.info(f"Sessões processadas: {len(result['processed_sessions'])}")
            if result['failed_sessions']:
                logger.warning(f"Sessões com erro: {len(result['failed_sessions'])}")
        else:
            logger.error(f"Erro: {result.get('error')}")
    
    # Caso contrário, processa sessão específica
    else:
        logger.info(f"Modo: Processamento de sessão específica")
        result = process_stt_results(args.session_dir)
        
        if result["success"]:
            logger.info("Processamento concluído com sucesso!")
            logger.info(f"Arquivos gerados: {len(result['output_files'])}")
        else:
            logger.error(f"Erro durante o processamento: {result.get('error')}")


if __name__ == "__main__":
    main()