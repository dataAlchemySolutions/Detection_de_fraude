# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 00:59:48 2025

@author: osahl
"""


import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

fraudTest= pd.read_csv("data/fraudTest.csv",index_col=0)
pred_test= pd.read_csv("app/predictions_fraudTest.csv",index_col=0)


y_test = fraudTest['is_fraud']
pred =  pred_test['fraud_prediction']


# Calcul des métriques sur l'ensemble de test
print("\n--- Évaluation sur l'ensemble de test ---")
precision_test = precision_score(y_test, pred)
recall_test = recall_score(y_test, pred)
f1_test = f1_score(y_test, pred)

# Affichage des métriques sur l'ensemble de test
print(f"Precision: {precision_test:.4f}")
print(f"Recall: {recall_test:.4f}")
print(f"F1-Score: {f1_test:.4f}")