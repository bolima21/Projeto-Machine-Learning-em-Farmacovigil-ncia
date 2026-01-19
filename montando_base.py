# -*- coding: utf-8 -*-

# Importação das bibliotecas necessárias
import pandas as pd
import numpy as np
from datetime import datetime

# Leitura dos arquivos CSV com o separador de ponto e vírgula e codificação 'latin-1'
# Carregar os dados de cada arquivo CSV
notifications = pd.read_csv('VigiMed_Notificacoes.csv', sep=';', encoding='latin-1', dtype=str)
reactions = pd.read_csv('VigiMed_Reacoes.csv', sep=';', encoding='latin-1', dtype=str)
# Tentar ler o arquivo ignorando as linhas ruins
try:
    medications = pd.read_csv(
        'Vigimed_Medicamentos.csv',
        sep=';',
        encoding='latin-1',
        engine='python',
        on_bad_lines='skip',
        dtype=str
    )
    print("Arquivo carregado com sucesso!")
except Exception as e:
    print(f"Ocorreu um erro: {e}")

# Exibir as primeiras linhas de cada DataFrame para entender as colunas disponíveis
topo_notificacoes = notifications.head()
topo_reacoes = reactions.head()
topo_medicamentos = medications.head()


# Selecionar as colunas desejadas do DataFrame reactions
columns_reactions = [
    'PT',
    'GRAVE',
    'IDENTIFICACAO_NOTIFICACAO'
]
reactions_selected = reactions[columns_reactions]

# Selecionar as colunas desejadas do DataFrame notifications
columns_notifications = [
    'UF',
    'TIPO_ENTRADA_VIGIMED',
    'RECEBIDO_DE',
    'IDENTIFICACAO_NOTIFICACAO',
    'DATA_NOTIFICACAO',
    'TIPO_NOTIFICACAO',
    'NOTIFICACAO_PARENT_CHILD',
    'IDADE_MOMENTO_REACAO',
    'IDADE_GESTACIONAL_MOMENTO_REACAO',
    'SEXO',
    'GESTANTE',
    'LACTANTE',
    'PESO_KG',
    'ALTURA_CM',
    'NOTIFICADOR'
]
notifications_selected = notifications[columns_notifications]

# Realizar o merge dos DataFrames reactions_selected e notifications_selected com base na coluna 'IDENTIFICACAO_NOTIFICACAO'
merged_df = pd.merge(reactions_selected, notifications_selected, on='IDENTIFICACAO_NOTIFICACAO', how='left')

# Selecionar as colunas desejadas do DataFrame medications
columns_medications = [
    'IDENTIFICACAO_NOTIFICACAO',
    'RELACAO_MEDICAMENTO_EVENTO',
    'PRINCIPIOS_ATIVOS_WHODRUG',
    'CONCENTRACAO',
    'DOSE',
    'FREQUENCIA_DOSE',
    'VIA_ADMINISTRACAO'
]
medications_selected = medications[columns_medications]

# Realizar o merge do DataFrame medications_selected com o DataFrame merged_df com base na coluna 'IDENTIFICACAO_NOTIFICACAO'
final_df = pd.merge(merged_df, medications_selected, on='IDENTIFICACAO_NOTIFICACAO', how='left')

# Selecionar as colunas finais para o DataFrame final
final_columns = [
    'PT',
    'GRAVE',
    'IDENTIFICACAO_NOTIFICACAO',
    'UF',
    'TIPO_ENTRADA_VIGIMED',
    'RECEBIDO_DE',
    'DATA_NOTIFICACAO',
    'TIPO_NOTIFICACAO',
    'NOTIFICACAO_PARENT_CHILD',
    'IDADE_MOMENTO_REACAO',
    'IDADE_GESTACIONAL_MOMENTO_REACAO',
    'SEXO',
    'GESTANTE',
    'LACTANTE',
    'PESO_KG',
    'ALTURA_CM',
    'NOTIFICADOR',
    'RELACAO_MEDICAMENTO_EVENTO',
    'PRINCIPIOS_ATIVOS_WHODRUG',
    'CONCENTRACAO',
    'DOSE',
    'FREQUENCIA_DOSE',
    'VIA_ADMINISTRACAO'
]
# Garantir que todas as colunas existam no DataFrame final
final_df = final_df[final_columns]

topo_final_df = final_df.head()

# Exportar o DataFrame final para um arquivo CSV
final_df.to_csv('base_reacoes_adversas.csv', index=False, sep=';', encoding='latin-1')

