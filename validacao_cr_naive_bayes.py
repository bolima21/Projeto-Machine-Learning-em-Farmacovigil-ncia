# -*- coding: utf-8 -*-

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

#VALIDAÇÃO CRUZADA
previsores = np.asarray(previsores)
classe = np.asarray(classe)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
acuracias = []
matrizes   = []
metricas   = []
y_true_all = []
y_score_all = []



print("Iniciando a validação cruzada para Naive Bayes...")
for fold, (idx_train, idx_test) in enumerate(kfold.split(previsores, classe), start=1):
    
    X_train, X_test = previsores[idx_train], previsores[idx_test]
    y_train, y_test = classe[idx_train], classe[idx_test]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    sampler = RandomUnderSampler(random_state=1)
    X_train_res, y_train_res = sampler.fit_resample(X_train_scaled, y_train)

    # Classificador
    classificador = GaussianNB()

    classificador.fit(X_train_res, y_train_res)
    
    previsoes = classificador.predict(X_test_scaled)
    
    acuracias.append(accuracy_score(y_test, previsoes))
    matrizes.append(confusion_matrix(y_test, previsoes))
    metricas.append(precision_recall_fscore_support(y_test, previsoes, zero_division=0))
    
    probas = classificador.predict_proba(X_test_scaled)[:, 1]
    y_true_all.extend(y_test)
    y_score_all.extend(probas)
    
    print(f"Fold {fold} — Acurácia: {acuracias[-1]:.4f}")

print("\nValidação cruzada concluída!")

# Resultados Agregados

print("\n--- Resultados Finais da Validação Cruzada (Naive Bayes) ---")

# Acurácia média e desvio
acuracia_final_media = np.mean(acuracias)
acuracia_final_desvio_padrao = np.std(acuracias)
print(f"\nAcurácia Média: {acuracia_final_media:.4f} ± {acuracia_final_desvio_padrao:.4f}")

# Matriz de confusão média
matriz_media = np.mean(matrizes, axis=0)
print("\nMatriz de Confusão Média:\n", matriz_media.round(1))
matriz_desvio = np.std(matrizes, axis=0)
print("\nDesvio Padrão da Matriz:\n", matriz_desvio.round(1))

print("\nMatriz de Confusão (Média ± Desvio Padrão): ")

label_vn = f"{matriz_media[0, 0]:.1f} ± {matriz_desvio[0, 0]:.1f}"
label_fp = f"{matriz_media[0, 1]:.1f} ± {matriz_desvio[0, 1]:.1f}"
label_fn = f"{matriz_media[1, 0]:.1f} ± {matriz_desvio[1, 0]:.1f}"
label_vp = f"{matriz_media[1, 1]:.1f} ± {matriz_desvio[1, 1]:.1f}"

print(f"[{label_vn}   {label_fp}]")
print(f"[{label_fn}   {label_vp}]")
# Métricas médias e desvios padrões por classe
precisao, recall, f1, suporte = zip(*metricas)
precisao_medias = np.mean(precisao, axis=0)
recall_medias   = np.mean(recall,   axis=0)
f1_medias       = np.mean(f1,       axis=0)
precisao_desvios = np.std(precisao, axis=0)
recall_desvios   = np.std(recall,   axis=0)
f1_desvios       = np.std(f1,       axis=0)

print("\n--- Métricas Médias por Classe ---")
print(f"Classe 0 (Não grave): Precisão={precisao_medias[0]:.4f} ± {precisao_desvios[0]:.4f}, "
      f"Recall={recall_medias[0]:.4f} ± {recall_desvios[0]:.4f}, F1-Score={f1_medias[0]:.4f} ± {f1_desvios[0]:.4f}")
print(f"Classe 1 (Grave): Precisão={precisao_medias[1]:.4f} ± {precisao_desvios[1]:.4f}, "
      f"Recall={recall_medias[1]:.4f} ± {recall_desvios[1]:.4f}, F1-Score={f1_medias[1]:.4f} ± {f1_desvios[1]:.4f}")

# Curva ROC e AUC Global
if len(y_true_all) > 0 and len(set(y_true_all)) == 2:
    fpr, tpr, thresholds = roc_curve(y_true_all, y_score_all)
    roc_auc = auc(fpr, tpr)
    print(f"\nAUC (ROC) Global: {roc_auc:.4f}")
    
    # Plotando o gráfico
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (área = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC da Validação Cruzada - Naive Bayes')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()