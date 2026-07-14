import re
import unicodedata
import pandas as pd

def clean_text(text):
    """
    Realiza a limpeza básica: minúsculas, remove espaços extras 
    e retira acentuação (normalização NFKD).
    """
    if pd.isna(text):
        return ""
    # Converte para minúsculas e remove espaços nas extremidades
    text = text.lower().strip()
    # Remove acentos
    text = unicodedata.normalize('NFKD', text)
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    return text

def remove_noise(text):
    """
    Remove padrões específicos que não agregam valor semântico à análise,
    como códigos de máquinas, nomes de operadores e parênteses vazios.
    """
    # Remove 'mip' seguido de números (ex: mip###, mip ####)
    #text = re.sub(r"mip\s*\d+", "", text)
    # Remove 'maquina ###', 'maq.###', etc.
    #text = re.sub(r"(maquina|maq\.?)\s*\d+", "", text)

    # O flags=re.IGNORECASE substitui a necessidade de colocar maiúsculas/minúsculas
    text = re.sub(r"(mip|maquina|maq\.?|maq)\s*\d+", "", text, flags=re.IGNORECASE)
    
    # Remove nomes próprios simples (ex: sr. lopes)
    text = re.sub(r"sr\.?\s*\w+", "", text, flags=re.IGNORECASE)
      
    # Remove parênteses vazios ou apenas com pontuação
    #text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\(\s*[^\w\s]*\s*\)", "", text)

    # Limpeza de resíduos de pontuação e espaços
    #text = text.strip().strip('-').strip('.').strip(':').strip()

    # Remove qualquer símbolo (hífen, ponto, dois-pontos, espaço) do INÍCIO
    text = re.sub(r"^[ \-\.\:\,\/]+", "", text)
    
    # Remove qualquer símbolo do FIM
    text = re.sub(r"[ \-\.\:\,\/]+$", "", text)

    # Normaliza múltiplos espaços para um único espaço
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()

def tag_event(text):
    """
    Classifica o evento com base em palavras-chave (Regra Semântica).
    Você pode expandir esta função conforme surgirem novas categorias.
    """
    if re.search(r"cotas", text):
        return "setup"
    if re.search(r"preventiv", text):
        return "preventiva"
    
    return "avaria"