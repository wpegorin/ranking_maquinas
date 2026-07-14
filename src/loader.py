import pandas as pd
import numpy as np
from .processing import clean_text, remove_noise, tag_event

def get_cleaned_data(filepath):
    """
    Carrega o arquivo Excel e aplica todo o pipeline de limpeza,
    filtros e cálculos de engenharia de manutenção.
    """
    
    # 1. Carga do Dataset
    # ---------------------------------------------------------
    df = pd.read_excel(filepath, sheet_name="work_orders")
    
    # 2. Filtros Iniciais (Manutenção Corretiva e Status)
    # ---------------------------------------------------------
    df = df[df["WORK_TYPE_DESC"] == "MANUTENCAO CORRECTIVA"].copy()
    
    # Filtramos apenas ordens finalizadas para garantir integridade dos cálculos de tempo
    status_validos = ['FINISHED', 'WORKDONE']
    #df = df[df['WO_STATUS_ID'].isin(status_validos)].copy()
    
    # 3. Tratamento de Texto 
    # ---------------------------------------------------------
    # Aplicamos as funções do processing.py em sequência
    df["ERR_DESCR_CLEAN"] = df["ERR_DESCR"].apply(clean_text).apply(remove_noise)
    df["EVENT_TAG"] = df["ERR_DESCR_CLEAN"].apply(tag_event)
    
    # 4. Conversão de Datas
    # ---------------------------------------------------------
    # Convertendo para datetime para permitir cálculos matemáticos
    df["REG_DATE_DT"] = pd.to_datetime(df["REG_DATE"])
    df["REAL_S_DATE_DT"] = pd.to_datetime(df["REAL_S_DATE"])
    df["REAL_F_DATE_DT"] = pd.to_datetime(df["REAL_F_DATE"])
    
    # 5. Lógica de Manutenção (TTR e Sobreposição)
    # ---------------------------------------------------------
    # Ordenação necessária para calcular o tempo entre falhas (Delta)
    df = df.sort_values(["MCH_CODE", "REG_DATE_DT"]).reset_index(drop=True)
      
    # Tempo entre falhas (Delta)
    df['delta_days'] = df.groupby('MCH_CODE')['REG_DATE_DT'].diff().dt.total_seconds() / (24 * 3600)
    
    # Identificar Próxima Falha para lógica de sobreposição
    df['PROXIMA_FALHA'] = df.groupby('MCH_CODE')['REG_DATE_DT'].shift(-1)
    
    # Flag de sobreposição: o reparo atual terminou depois que a próxima falha começou?
    df['SOBREPOSICAO_REPARO'] = df['REAL_F_DATE_DT'] > df['PROXIMA_FALHA']
    
    # Ajuste para o último evento de cada máquina
    df.loc[df['PROXIMA_FALHA'].isna(), 'SOBREPOSICAO_REPARO'] = False

    # Tempo para Reparo (TTR) em dias
    df['TTR_dias'] = (df['REAL_F_DATE_DT'] - df['REAL_S_DATE_DT']).dt.total_seconds() / (24 * 3600)

    # Filtramos apenas ordens finalizadas após cálculos de tempo, pois a Work Order não deve ser sobreposta.
    df = df[df['WO_STATUS_ID'].isin(status_validos)].copy()
       
    return df
