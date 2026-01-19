# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

base = pd.read_csv('base_reacoes_adversas.csv', sep=';', encoding='latin-1', dtype=str)

if 'DATA_NOTIFICACAO' in base.columns:
    base['DATA_NOTIFICACAO'] = pd.to_datetime(base['DATA_NOTIFICACAO'], errors='coerce')
    base['ANO_NOTIFICACAO'] = base['DATA_NOTIFICACAO'].dt.year
    base = base.drop(columns=['DATA_NOTIFICACAO'])
    print("Coluna 'ANO_NOTIFICACAO' criada e 'DATA_NOTIFICACAO' removida.")
else:
    print("A coluna 'DATA_NOTIFICACAO' não está presente no DataFrame e será ignorada.")

def converter_idade_para_anos(valor):
    if pd.isnull(valor):
        return np.nan
    valor = str(valor).strip().upper()
    match = re.match(r'(\d+\.?\d*)\s*([DMA])?', valor)
    if not match:
        return np.nan
    numero = float(match.group(1))
    unidade = match.group(2)
    if unidade == 'D':
        return numero / 365.25
    elif unidade == 'M':
        return numero / 12
    elif unidade == 'A' or unidade is None:
        return numero
    else:
        return np.nan

if 'IDADE_MOMENTO_REACAO' in base.columns:
    base['IDADE_MOMENTO_REACAO'] = base['IDADE_MOMENTO_REACAO'].apply(converter_idade_para_anos)

def extrair_numero(texto):
    if pd.isnull(texto):
        return np.nan
    resultado = re.findall(r'\d+\.?\d*', str(texto).replace(',', '.'))
    return float(resultado[0]) if resultado else np.nan

colunas_para_limpar = [
    'IDADE_MOMENTO_REACAO',
    'IDADE_GESTACIONAL_MOMENTO_REACAO',
    'PESO_KG',
    'ALTURA_CM',
    'CONCENTRACAO',
    'DOSE'
]

for coluna in colunas_para_limpar:
    if coluna in base.columns:
        base[coluna] = base[coluna].apply(extrair_numero)

base = base.dropna(subset=['GRAVE'])

print(f"Linhas antes da remoção de vazamento: {len(base)}")

leaker_1 = base['PRINCIPIOS_ATIVOS_WHODRUG'].str.contains('Eculizumab', case=False, na=False)
leaker_2 = base['PT'].str.contains('Neutropenia', case=False, na=False)

total_leakers = leaker_1 | leaker_2
base = base[~total_leakers]

print(f"Linhas após a remoção de vazamento: {len(base)} ({total_leakers.sum()} linhas removidas)")

base = base.sample(frac=0.2, random_state=42)

print(f"\nBase reduzida para {base.shape[0]} linhas (amostragem).")

colunas_para_verificar_nulos = ['DOSE', 'VIA_ADMINISTRACAO', 'ALTURA_CM']
colunas_existentes_nulos = [col for col in colunas_para_verificar_nulos if col in base.columns]
base = base.dropna(subset=colunas_existentes_nulos)

print("\n=== ESTATÍSTICAS RELATIVAS À BASE ===")

qtd_final = len(base)
contagem_grave = base['GRAVE'].value_counts()
porcentagem_grave = base['GRAVE'].value_counts(normalize=True) * 100

ano_inicial = "Indefinido"
ano_final = "Indefinido"

if 'ANO_NOTIFICACAO' in base.columns:
    anos_reais = base['ANO_NOTIFICACAO'][base['ANO_NOTIFICACAO'] <= 2025]
    if not anos_reais.empty:
        ano_inicial = int(anos_reais.min())
        ano_final = int(anos_reais.max())
    else:
        ano_inicial = int(base['ANO_NOTIFICACAO'].min())
        ano_final = int(base['ANO_NOTIFICACAO'].max())

print(f"Total de linhas finais: {qtd_final}")
print(f"Intervalo de anos: {ano_inicial} a {ano_final}")
print("\nProporção da Classe GRAVE:")
print(f"Grave (Sim): {contagem_grave.get('Sim', 0)} ({porcentagem_grave.get('Sim', 0):.2f}%)")
print(f"Não Grave (Não): {contagem_grave.get('Não', 0)} ({porcentagem_grave.get('Não', 0):.2f}%)")
print("========================================\n")

percentual_nulos = base.isnull().mean() * 100
colunas_para_remover_nulos = percentual_nulos[percentual_nulos > 50].index
base = base.drop(columns=colunas_para_remover_nulos)

colunas_numericas = [
    'ANO_NOTIFICACAO',
    'IDADE_MOMENTO_REACAO',
    'IDADE_GESTACIONAL_MOMENTO_REACAO',
    'PESO_KG',
    'ALTURA_CM',
    'CONCENTRACAO',
    'DOSE'
]

colunas_numericas = [col for col in colunas_numericas if col in base.columns]

colunas_categoricas = [
    col for col in base.columns
    if col not in colunas_numericas + ['GRAVE', 'IDENTIFICACAO_NOTIFICACAO']
]

for coluna in colunas_numericas:
    media = base[coluna].mean()
    base[coluna].fillna(media, inplace=True)

for coluna in colunas_categoricas:
    moda = base[coluna].mode()[0]
    base[coluna].fillna(moda, inplace=True)

if 'ALTURA_CM' in base.columns:
    base = base[(base['ALTURA_CM'] >= 50) & (base['ALTURA_CM'] <= 250)]

if 'PESO_KG' in base.columns:
    base = base[(base['PESO_KG'] >= 1) & (base['PESO_KG'] <= 300)]

if 'ALTURA_CM' in base.columns:
    base.loc[base['ALTURA_CM'] < 3, 'ALTURA_CM'] *= 100

if 'PESO_KG' in base.columns:
    base.loc[base['PESO_KG'] > 300, 'PESO_KG'] /= 1000

print("========================================\n")
print("FIM DA ANALISE EXPLORATORIA!!")
print("========================================\n")

for coluna in colunas_categoricas:
    top_categorias = base[coluna].value_counts().nlargest(10).index
    base[coluna] = np.where(base[coluna].isin(top_categorias), base[coluna], 'OUTROS')

base = pd.get_dummies(base, columns=colunas_categoricas, drop_first=True)

classe = base['GRAVE']
le = LabelEncoder()
classe = le.fit_transform(classe)

previsores = base.drop(['GRAVE', 'IDENTIFICACAO_NOTIFICACAO'], axis=1, errors='ignore')
