#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Normalizador de Texto para Testes de Similaridade
# Baseado nos scripts text_normalization.py e transcription_normalizer.py
# Filosofia KISS - simples e funcional
#

import os
import re
import json
import unicodedata
from datetime import datetime
from pathlib import Path

# Mapeamento de caracteres especiais para português
chars_map = {
    'ï': 'i', 'ù': 'u', 'ö': 'o', 'î': 'i', 'ñ': 'n', 
    'ë': 'e', 'ì': 'i', 'ò': 'o', 'ů': 'u', 'ẽ': 'e', 
    'ü': 'u', 'è': 'e', 'æ': 'a', 'å': 'a', 'ø': 'o',
    'þ': 't', 'ð': 'd', 'ß': 's', 'ł': 'l', 'đ': 'd',
    'ć': 'c', 'č': 'c', 'š': 's', 'ž': 'z', 'ý': 'y'
}

def apply_char_mapping(text):
    """
    Aplica mapeamento de caracteres especiais
    """
    for special_char, normal_char in chars_map.items():
        text = text.replace(special_char, normal_char)
    return text
try:
    from utils.number_to_text import number_to_text
    HAS_NUMBER_TO_TEXT = True
except ImportError:
    HAS_NUMBER_TO_TEXT = False
    print("Módulo utils.number_to_text não encontrado. Usando conversão simples.")

def number_to_words_pt(num):
    """
    Converte número para extenso em português
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
            result += " e " + number_to_words_pt(remainder)
        
        return result
    
    # Para números maiores, retorna o número original
    return str(num)

def ordinal_to_words_pt(num, gender='m'):
    """
    Converte número ordinal para extenso em português
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
    
    # Para números não mapeados, usa o cardinal + "º/ª"
    cardinal = number_to_words_pt(num)
    suffix = "ª" if gender == 'f' else "º"
    return f"{cardinal}{suffix}"

def advanced_number_to_text(text):
    """
    Conversão avançada de números e símbolos para texto
    """
    result = text
    
    # Primeiro, trata ordinais (1º, 2ª, 15º, etc.)
    def replace_ordinal(match):
        num = int(match.group(1))
        suffix = match.group(2)
        gender = 'f' if suffix in ['ª', 'a'] else 'm'
        return ordinal_to_words_pt(num, gender)
    
    # Regex para ordinais: 1º, 2ª, 15º, etc.
    result = re.sub(r'(\d+)([ºª°])', replace_ordinal, result)
    
    # Trata números decimais (ex: 20,50 ou 1.5)
    def replace_decimal(match):
        full_match = match.group(0)
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

def remove_html_tags(text):
    """
    Remove tags HTML usando regex
    """
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def text_cleaning(text):
    """
    Limpeza e normalização de texto baseada nos scripts originais
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

def normalize_text(text):
    """
    Normalização completa do texto
    """
    if not text or text.strip() == "":
        return None
    
    # Converte números para texto
    if HAS_NUMBER_TO_TEXT:
        try:
            text = number_to_text(text)
        except Exception as e:
            print(f"Erro na conversão de números: {e}")
            text = advanced_number_to_text(text)
    else:
        text = advanced_number_to_text(text)
    
    # Aplica limpeza
    normalized = text_cleaning(text)
    
    return normalized if normalized else None

def extract_file_info(filename):
    """
    Extrai informações do nome do arquivo dinamicamente
    Padrão: {video_id}_..._segment_{000}_.._{modelo}.txt
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

def read_text_file(filepath):
    """
    Lê arquivo de texto com tratamento de encoding
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
            print(f"Erro ao ler {filepath}: {e}")
            return None
    except Exception as e:
        print(f"Erro ao ler {filepath}: {e}")
        return None

def process_folder(folder_path):
    """
    Processa uma pasta com arquivos de transcrição
    """
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"Pasta não encontrada: {folder_path}")
        return False
    
    print(f"Processando pasta: {folder_path}")
    
    # Busca todos os arquivos .txt
    txt_files = list(folder_path.glob("*.txt"))
    
    if not txt_files:
        print("Nenhum arquivo .txt encontrado na pasta")
        return False
    
    print(f"Encontrados {len(txt_files)} arquivos .txt")
    
    # Agrupa arquivos por video_id e segment
    grouped_files = {}
    
    for txt_file in txt_files:
        video_id, segment_number, modelo = extract_file_info(txt_file.name)
        
        if not all([video_id, segment_number, modelo]):
            print(f"Erro ao extrair informações de: {txt_file.name}")
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
        print("Nenhum arquivo válido processado")
        return False
    
    # Agrupa por video_id para criar JSONs separados
    videos = {}
    for key, data in grouped_files.items():
        video_id = data['video_id']
        if video_id not in videos:
            videos[video_id] = {}
        videos[video_id][key] = data
    
    # Cria JSON para cada video_id
    for video_id, segments in videos.items():
        normalized_pairs = {}
        valid_pairs = 0
        
        # Ordena segmentos por número (0000, 0001, 0002, etc.)
        sorted_segments = sorted(segments.items(), key=lambda x: x[0])
        
        for segment_key, data in sorted_segments:
            # Verifica se tem ambos os modelos
            has_wav2vec2 = 'wav2vec2_original' in data
            has_whisper = 'whisper_original' in data
            
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
        
        # Salva JSON
        output_file = folder_path / f"{video_id}_normalized_text.json"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"Arquivo salvo: {output_file}")
            print(f"Video ID: {video_id}")
            print(f"Total de segmentos: {len(normalized_pairs)}")
            print(f"Pares válidos: {valid_pairs}")
            print("-" * 50)
            
        except Exception as e:
            print(f"Erro ao salvar {output_file}: {e}")
            return False
    
    return True

def main():
    """
    Função principal
    """
    print("NORMALIZADOR DE TEXTO PARA SIMILARIDADE")
    print("=" * 50)
    
    # Solicita caminho da pasta (compatível com qualquer SO)
    folder_input = input("Digite o caminho da pasta com os arquivos .txt: ").strip()
    
    if not folder_input:
        print("Caminho não fornecido. Saindo...")
        return
    
    # Normaliza o caminho para o SO atual
    folder_path = str(Path(folder_input).resolve())
    
    if not folder_path:
        print("Caminho não fornecido. Saindo...")
        return
    
    # Processa a pasta
    success = process_folder(folder_path)
    
    if success:
        print("Processamento concluído com sucesso!")
    else:
        print("Erro durante o processamento.")

if __name__ == "__main__":
    main()