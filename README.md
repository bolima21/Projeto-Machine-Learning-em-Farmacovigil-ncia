# Projeto-Machine-Learning-em-Farmacovigil-ncia
**Predição de Gravidade de Eventos Adversos em Farmacovigilância (VigiMed)**

Este repositório contém os códigos fonte e experimentos desenvolvidos como parte de um Projeto Acadêmico para o curso de Bacharelado em Sistemas de Informação.

**Sobre o Projeto:**
A farmacovigilância é essencial para monitorar a segurança de medicamentos pós-comercialização. 
Este projeto visa aplicar técnicas de Machine Learning para predizer a classificação de gravidade ("Grave" ou "Não Grave") de Reações Adversas a Medicamentos (RAMs), utilizando dados reais do sistema VigiMed da ANVISA.

O objetivo é comparar o desempenho de diferentes algoritmos de classificação supervisionada e identificar quais atributos (como frequência de dose, tipo de notificador, fármaco etc) 
mais influenciam na determinação da gravidade de um evento.

**Principais Funcionalidades:**
-**Consolidação**: 3 bases do sistema VigiMed (VigiMed_Medicamentos, VigiMed_Notificacoes e VigiMed_Reacoes) transformadas em uma base única, por meio de uma variável de convergência e a seleção de atributos mais relevantes
-**Pré-processamento**:  Conversão de dados, tratamento de valores nulos/ausentes, tratamento de vazamentos de dados (data leakage), codificação de variáveis previsoras categóricas (one-hot-encoding) e transformação da
variável objetivo (label-encoding).
-**Balanceamento**: Aplicação de Undersampling

**Técnicas de Machine Learning utilizadas:**
-K-Nearest Neighbors (KNN)
-Naive Bayes (Gaussian)
-Decision Tree
-Random Forest
-Extreme Gradient Boosting (XGBoost)
-Support Vector Machine (SVM)
-Redes Neurais (MLP)

**Métricas de Avaliação (Validação Cruzada Estratificada):**
-Matriz de confusão
-Acurácia
-Precisão
-Recall
-F1-Score
-AUC (ROC)

**Tecnologias Utilizadas:**
-Linguagem: Python 3.11.7
-IDE: Spyder 5.4.3
-Bibliotecas utilizadas: Pandas, Scikit-Learn, NumPy, Matplotlib, XGBoost

**Estrutura do Repositório:**
- montando_base.py: Script responsável por ler os arquivos CSV originais da ANVISA e consolidá-los em um único dataset (base_reacoes_adversas.csv).
- pre_processamento_atual.py: Realiza a limpeza dos dados, remoção de features com vazamento (Eculizumab, Neutropenia), tratamento de valores nulos e codificação.
- Scripts de Validação (validacao_cr_*.py):
    * Cada arquivo contém a implementação de um algoritmo específico (ex.: validacao_cr_xgboost.py, validacao_cr_knn.py).
    * Executam a validação cruzada, geram matrizes de confusão e curvas ROC.
 
**Estado Atual:**
O Projeto está passando por um processo de refatoração, buscando reduzir a repetição de códigos, bem como aprimorar as configurações dos hiperparâmetros utilizados, com o intuito de obter resultados melhores.


**Como executar o Projeto:**
**1 - Obtenção dos Dados (Obrigatório):** Devido ao volume de dados, os arquivos originais não estão incluídos neste repositório. Para executar os scripts, siga os passos abaixo:

  1.1 - Acesse o Portal de Dados Abertos da ANVISA: https://dados.gov.br/dados/conjuntos-dados/notificacoes-em-farmacovigilancia

  1.2 - Baixe os arquivos CSV mais recentes referentes a: Notificações (VigiMed_Notificacoes.csv), Medicamentos (VigiMed_Medicamentos.csv) e Reações (VigiMed_Reacoes.csv).

  1.3 - Caso os arquivos não tenham os nomes acima, renomeie-os conforme indicado anteriormente.

**2 - Instalação das Dependências:** Certifique-se de ter a Linguagem Python instalada e execute o comando abaixo para instalar as bibliotecas necessárias:
pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn

**3 - Ordem de Execução dos scripts:**
  3.1 - Execute o script montando_base.py. Ele irá ler os três arquivos CSV baixados, processar as chaves de identificação e gerar um arquivo consolidado chamado base_reacoes_adversas.csv.
  3.2 - Com a base consolidada gerada, execute o script pre_processamento_atual.py para carregar as variáveis em memória.
  3.3 - Execute qualquer um dos scripts de validação (ex: validacao_cr_knn.py) para treinar o modelo específico e visualizar as métricas de desempenho.
