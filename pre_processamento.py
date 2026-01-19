# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder


def tratar_idade(valor):
    
    #Converte idade informada em dias, meses ou anos para anos (float).
    #Exemplos: 30D, 10M, 5A.
    
    if pd.isnull(valor):
        return np.nan

    valor = str(valor).strip().upper()
    correspondencia = re.match(r'(\d+\.?\d*)\s*([DMA])?', valor)

    if not correspondencia:
        return np.nan

    numero = float(correspondencia.group(1))
    unidade = correspondencia.group(2)

    if unidade == 'D':
        return numero / 365.25
    if unidade == 'M':
        return numero / 12

    return numero  # Anos ou sem unidade explícita


def limpar_numeros(valor):
    
    #Extrai e padroniza valores numéricos a partir de campos textuais.
    
    if pd.isnull(valor):
        return np.nan

    numeros = re.findall(r'\d+\.?\d*', str(valor).replace(',', '.'))
    return float(numeros[0]) if numeros else np.nan


def main():
    # Carregamento da base
    print("Carregando base...")
    df_base = pd.read_csv(
        'base_reacoes_adversas.csv',
        sep=';',
        encoding='latin-1',
        dtype=str
    )

    # Ajuste de data para extração do ano de notificação
    if 'DATA_NOTIFICACAO' in df_base.columns:
        df_base['DATA_NOTIFICACAO'] = pd.to_datetime(
            df_base['DATA_NOTIFICACAO'],
            errors='coerce'
        )
        df_base['ANO_NOTIFICACAO'] = df_base['DATA_NOTIFICACAO'].dt.year
        df_base = df_base.drop(columns=['DATA_NOTIFICACAO'])

    # Limpeza e padronização de idade
    if 'IDADE_MOMENTO_REACAO' in df_base.columns:
        df_base['IDADE_MOMENTO_REACAO'] = (
            df_base['IDADE_MOMENTO_REACAO']
            .apply(tratar_idade)
        )

    colunas_numericas_brutas = [
        'IDADE_GESTACIONAL_MOMENTO_REACAO',
        'PESO_KG',
        'ALTURA_CM',
        'CONCENTRACAO',
        'DOSE'
    ]

    for coluna in colunas_numericas_brutas:
        if coluna in df_base.columns:
            df_base[coluna] = df_base[coluna].apply(limpar_numeros)

    # Remoção de registros sem variável alvo (prevenção de vazamento)
    df_base = df_base.dropna(subset=['GRAVE'])
    print(f"Total antes da remoção de vazamento: {len(df_base)}")

    filtro_eculizumabe = df_base['PRINCIPIOS_ATIVOS_WHODRUG'].str.contains(
        'Eculizumab', case=False, na=False
    )
    filtro_neutropenia = df_base['PT'].str.contains(
        'Neutropenia', case=False, na=False
    )

    df_base = df_base[~(filtro_eculizumabe | filtro_neutropenia)]

    print(
        f"Total após remoção de vazamento: {len(df_base)} "
        f"({(filtro_eculizumabe | filtro_neutropenia).sum()} excluídos)"
    )

    # Amostragem para redução de dimensionalidade
    df_base = df_base.sample(frac=0.2, random_state=42)
    print(f"Base reduzida (20%): {len(df_base)} registros.")

    # Remoção de valores ausentes em colunas críticas
    colunas_criticas = ['DOSE', 'VIA_ADMINISTRACAO', 'ALTURA_CM']
    colunas_existentes = [c for c in colunas_criticas if c in df_base.columns]
    df_base = df_base.dropna(subset=colunas_existentes)

    # Estatísticas básicas da base processada
    total_registros = len(df_base)
    contagem_gravidade = df_base['GRAVE'].value_counts()
    percentual_gravidade = df_base['GRAVE'].value_counts(normalize=True) * 100

    ano_inicio, ano_fim = "Indefinido", "Indefinido"
    if 'ANO_NOTIFICACAO' in df_base.columns:
        anos_validos = df_base['ANO_NOTIFICACAO'][
            df_base['ANO_NOTIFICACAO'] <= 2025
        ]
        if not anos_validos.empty:
            ano_inicio = int(anos_validos.min())
            ano_fim = int(anos_validos.max())

    print("\n=== ESTATÍSTICAS DA BASE PROCESSADA ===")
    print(f"Total de linhas finais: {total_registros}")
    print(f"Período dos dados: {ano_inicio} a {ano_fim}")
    print(
        f"Grave: {contagem_gravidade.get('Sim', 0)} "
        f"({percentual_gravidade.get('Sim', 0):.2f}%)"
    )
    print(
        f"Não Grave: {contagem_gravidade.get('Não', 0)} "
        f"({percentual_gravidade.get('Não', 0):.2f}%)"
    )
    print("=======================================\n")

    # Tratamento de valores ausentes e outliers
    percentual_nulos = df_base.isnull().mean()
    df_base = df_base.drop(
        columns=percentual_nulos[percentual_nulos > 0.5].index
    )

    colunas_numericas = [
        'ANO_NOTIFICACAO',
        'IDADE_MOMENTO_REACAO',
        'IDADE_GESTACIONAL_MOMENTO_REACAO',
        'PESO_KG',
        'ALTURA_CM',
        'CONCENTRACAO',
        'DOSE'
    ]
    colunas_numericas = [
        c for c in colunas_numericas if c in df_base.columns
    ]

    colunas_categoricas = [
        c for c in df_base.columns
        if c not in colunas_numericas + ['GRAVE', 'IDENTIFICACAO_NOTIFICACAO']
    ]

    # Imputação
    for coluna in colunas_numericas:
        df_base[coluna].fillna(df_base[coluna].mean(), inplace=True)

    for coluna in colunas_categoricas:
        if not df_base[coluna].dropna().empty:
            df_base[coluna].fillna(df_base[coluna].mode()[0], inplace=True)

    # Correções de escala e remoção de outliers fisiologicamente inválidos
    if 'ALTURA_CM' in df_base.columns:
        df_base = df_base[
            (df_base['ALTURA_CM'] >= 50) &
            (df_base['ALTURA_CM'] <= 250)
        ]
        df_base.loc[df_base['ALTURA_CM'] < 3, 'ALTURA_CM'] *= 100

    if 'PESO_KG' in df_base.columns:
        df_base = df_base[
            (df_base['PESO_KG'] >= 1) &
            (df_base['PESO_KG'] <= 300)
        ]
        df_base.loc[df_base['PESO_KG'] > 300, 'PESO_KG'] /= 1000

    # Codificação categórica (One-Hot Encoding com limitação de cardinalidade)
    for coluna in colunas_categoricas:
        categorias_top10 = df_base[coluna].value_counts().nlargest(10).index
        df_base[coluna] = np.where(
            df_base[coluna].isin(categorias_top10),
            df_base[coluna],
            'OUTROS'
        )

    df_base = pd.get_dummies(
        df_base,
        columns=colunas_categoricas,
        drop_first=True
    )

    # Saída final para modelagem
    label_encoder = LabelEncoder()
    classe = label_encoder.fit_transform(df_base['GRAVE'])

    previsores = df_base.drop(
        ['GRAVE', 'IDENTIFICACAO_NOTIFICACAO'],
        axis=1,
        errors='ignore'
    )

    print("Pré-processamento concluído.")
    return previsores, classe


if __name__ == "__main__":
    main()
