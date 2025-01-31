# -*- coding: utf-8 -*-


# modèles de détection des fraudes pour identifier les transactions potentiellement frauduleuses.
# Détection de Fraude (Classification)
# Utilisation de modèles supervisés pour identifier des transactions suspectes dans les banques, 
# les assurances ou les plateformes de e-commerce.
# Algorithmes : Random Forest, XGBoost, Deep Learning.


# https://www.kaggle.com/datasets/kartik2112/fraud-detection?


import pandas as pd
from sklearn.utils import shuffle

# Lire le fichier CSV
fraudTrain= pd.read_csv("data/fraudTrain.csv",index_col=0)
fraudTest= pd.read_csv("data/fraudTest.csv",index_col=0)

data = pd.concat([fraudTrain, fraudTest], ignore_index=True)


data = shuffle(data, random_state=42)


# Dictionnaire pour renommer les colonnes
rename_columns = {

    
    "merchant": "Nom_marchand",
    "category": "Categorie_marchand",
    "first": "Prenom_titulaire",
    "last": "Nom_titulaire",
    "street": "Adresse_titulaire",
    "city": "Ville_titulaire",
    "state": "Etat_titulaire",
    "job": "Emploi_titulaire",
    "gender": "Sexe_titulaire",
    
    
    "trans_date_trans_time": "DateHeure_transaction",
    "unix_time": "Heure_UNIX_transaction",
    
    "merch_lat": "Latitude_marchand",
    "merch_long": "Longitude_marchand",
    "is_fraud": "Indicateur_fraude",
    "lat": "Latitude_titulaire",
    "long": "Longitude_titulaire",
    
    
    
    "index": "Identificateur_unique",
    "cc_num": "Num_carte_credit",
    "trans_num": "Numero_transaction",
    
    "amt": "Montant_transaction",
    "city_pop": "Population_ville",
    
    "zip": "Code_postal",
 
    "dob": "Date_naissance_titulaire"
    

}

# Renommer les colonnes
data.rename(columns=rename_columns, inplace=True)




#---------------  Vérification des valeurs manquantes  -----------------------#

# Vérifier les valeurs manquantes
missing_values_data = data.isnull().sum()

# Afficher le nombre de valeurs manquantes par colonne
print("Valeurs manquantes dans le jeu :")
print(missing_values_data)





#---------------  Vérification des doublons -----------------------#


# Vérifier les doublons dans les jeux de données
duplicates_data = data.duplicated().sum()

print(f"Nombre de doublons dans le jeu : {duplicates_data}")

# Si des doublons sont présents, tu peux les supprimer
data.drop_duplicates(inplace=True)







#---------------   Vérification des incohérences dans les données-----------------------#

# Vérifier les transactions avec un montant négatif ou nul
data = data[data['Montant_transaction'] > 0]

# Vérifier les dates de naissance incohérentes (par exemple, des personnes ayant plus de 120 ans)
data['Date_naissance_titulaire'] = pd.to_datetime(data['Date_naissance_titulaire'], errors='coerce')
data = data[data['Date_naissance_titulaire'] > pd.Timestamp.today() - pd.Timedelta(days=120*365)]

# Vérifier les coordonnées géographiques valides
data = data[(data['Latitude_titulaire'] >= -90) & (data['Latitude_titulaire'] <= 90)]
data = data[(data['Longitude_titulaire'] >= -180) & (data['Longitude_titulaire'] <= 180)]

# Vérifier les coordonnées géographiques valides pour Latitude_marchand et Longitude_marchand
data = data[(data['Latitude_marchand'] >= -90) & (data['Latitude_marchand'] <= 90)]
data = data[(data['Longitude_marchand'] >= -180) & (data['Longitude_marchand'] <= 180)]




#---------------   Encodage des variables catégorielles-----------------------#

# Target Encoding pour Nom_marchand
data['Nom_marchand_encoded']= data.groupby('Nom_marchand')['Indicateur_fraude'].transform('mean')
#Sauvegarde
nom_marchand_means = data.groupby('Nom_marchand')['Indicateur_fraude'].mean()


data = data.drop(['Nom_marchand'], axis=1)

data = data.drop(['Prenom_titulaire', 'Nom_titulaire'], axis=1)

data = pd.get_dummies(data, columns=['Categorie_marchand'], drop_first=True)

# Convertir les colonnes booléennes en valeurs 0 et 1
data = data.astype({col: 'int' for col in data.select_dtypes(include=['bool']).columns})






from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data['Sexe_titulaire'] = le.fit_transform(data['Sexe_titulaire'])



data = data.drop(['Adresse_titulaire'], axis=1)

data['Ville_titulaire_encoded'] = data.groupby('Ville_titulaire')['Indicateur_fraude'].transform('mean')
#Sauvegarde 
ville_titulaire_means = data.groupby('Ville_titulaire')['Indicateur_fraude'].mean()


data['Etat_titulaire_encoded'] = data.groupby('Etat_titulaire')['Indicateur_fraude'].transform('mean')
#♦sauvegarde
etat_titulaire_means = data.groupby('Etat_titulaire')['Indicateur_fraude'].mean()


data['Emploi_titulaire_encoded'] = data.groupby('Emploi_titulaire')['Indicateur_fraude'].transform('mean')
#Sauvegarde 
emploi_titulaire_means = data.groupby('Emploi_titulaire')['Indicateur_fraude'].mean()

data = data.drop(['Ville_titulaire', 'Etat_titulaire', 'Emploi_titulaire'], axis=1)

# Assurer que la colonne Date_naissance_titulaire est au bon format datetime
data['Date_naissance_titulaire'] = pd.to_datetime(data['Date_naissance_titulaire'], errors='coerce')

# Calculer l'âge en années en utilisant la différence avec la date actuelle
data['Age_titulaire'] = (pd.Timestamp.today() - data['Date_naissance_titulaire']).dt.days // 365

data = data.drop(['Date_naissance_titulaire'], axis=1)



#---------------  Transformation des variables temporelles  -----------------------#

# Assure-toi que la colonne 'DateHeure_transaction' est bien au format datetime
data['DateHeure_transaction'] = pd.to_datetime(data['DateHeure_transaction'])

# Extraire des informations temporelles
data['jour_semaine'] = data['DateHeure_transaction'].dt.weekday  # Lundi = 0, Dimanche = 6

# Est-ce un week-end (1 = week-end, 0 = jour de semaine)
data['weekend'] = data['jour_semaine'].apply(lambda x: 1 if x >= 5 else 0)

# Heure de la journée (extraction de l'heure)
data['heure_jour'] = data['DateHeure_transaction'].dt.hour

# Mois (permet d'analyser les tendances mensuelles)
data['mois'] = data['DateHeure_transaction'].dt.month

# Année (utile si tu veux vérifier les tendances sur plusieurs années)
data['annee'] = data['DateHeure_transaction'].dt.year

# Optionnel : Jour du mois (si tu veux explorer des tendances spécifiques à des jours particuliers)
data['jour_du_mois'] = data['DateHeure_transaction'].dt.day

# Supprimer les colonnes non utiles
data = data.drop(columns=['DateHeure_transaction'])

# Supprimer la colonne Heure_UNIX_transaction
data.drop(columns=['Heure_UNIX_transaction'], inplace=True)

# Supprimer les colonnes non utiles
data = data.drop(columns=['Num_carte_credit', 'Numero_transaction'])

# Target encoding pour le Code_postal (moyenne de la cible 'Indicateur_fraude')
data['Code_postal_target'] = data.groupby('Code_postal')['Indicateur_fraude'].transform('mean')
#Sauvegarde
code_postal_means = data.groupby('Code_postal')['Indicateur_fraude'].mean()

# Supprimer les colonnes non utiles
data = data.drop(columns=['Code_postal'])

# Vérifier les types de données dans data
print("Types de données dans data :")
print(data.dtypes)





#---------------  Vérification de la balance des classes  -----------------------#


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

# Vérification de la balance des classes dans le jeu de données
print(data['Indicateur_fraude'].value_counts())



X = data.drop(columns=['Indicateur_fraude'])

# Liste des colonnes dans l'ordre souhaité
columns_order = ['Montant_transaction', 'Sexe_titulaire', 'Latitude_titulaire',
                 'Longitude_titulaire', 'Population_ville', 'Latitude_marchand',
                 'Longitude_marchand', 'Nom_marchand_encoded',
                 'Categorie_marchand_food_dining', 'Categorie_marchand_gas_transport',
                 'Categorie_marchand_grocery_net', 'Categorie_marchand_grocery_pos',
                 'Categorie_marchand_health_fitness', 'Categorie_marchand_home',
                 'Categorie_marchand_kids_pets', 'Categorie_marchand_misc_net',
                 'Categorie_marchand_misc_pos', 'Categorie_marchand_personal_care',
                 'Categorie_marchand_shopping_net', 'Categorie_marchand_shopping_pos',
                 'Categorie_marchand_travel', 'Ville_titulaire_encoded',
                 'Etat_titulaire_encoded', 'Emploi_titulaire_encoded', 'Age_titulaire',
                 'jour_semaine', 'weekend', 'heure_jour', 'mois', 'annee',
                 'jour_du_mois', 'Code_postal_target']

# Réorganiser les colonnes du DataFrame X
X = X[columns_order]




y = data['Indicateur_fraude']


import matplotlib.pyplot as plt

# Création de l'histogramme
plt.figure(figsize=(8, 6))
plt.hist(y, bins=2, edgecolor='black', alpha=0.7)

# Ajout des labels et du titre
plt.xlabel('Valeur de Indicateur_fraude')
plt.ylabel('Fréquence')
plt.title('Histogramme de l\'Indicateur_fraude')

# Affichage de l'histogramme
plt.xticks([0, 1])
plt.show()




"""

from sklearn.preprocessing import StandardScaler
# Créer un objet StandardScaler
scaler = StandardScaler()
# Sélectionner uniquement les colonnes numériques
numerical_columns = X.select_dtypes(include=['float64', 'int64','int32']).columns
# Appliquer la normalisation
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

"""

# Séparation de l'ensemble d'entraînement et de test (80%/20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




#--------------- Modèle ---------------#



from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC



# Configuration de la validation croisée
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Modèle
#rf_model = RandomForestClassifier(random_state=42)

# Remplace le modèle RandomForest par XGBoost
rf_model = xgb.XGBClassifier(random_state=42)

#rf_model = lgb.LGBMClassifier(random_state=42)

#rf_model = LogisticRegression(random_state=42)

#rf_model = SVC(random_state=42)

#--------------- Validation croisée  ---------------#

# SMOTE pour équilibrer les classes dans les données d'entraînement
#smote = SMOTE(random_state=42)

metrics = {
    'precision': [],
    'recall': [],
    'f1_score': []
}


# Validation croisée avec SMOTE dans chaque fold
for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    print(f"\nFold {fold + 1}")
    
    # Séparer les données en train et validation
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # Appliquer SMOTE uniquement sur les données d'entraînement
    #X_train_fold_smote, y_train_fold_smote = smote.fit_resample(X_train_fold, y_train_fold)
    
    # Entraîner le modèle sur les données SMOTE du fold
    #rf_model.fit(X_train_fold_smote, y_train_fold_smote)
    
    rf_model.fit(X_train_fold, y_train_fold)
    
    # Prédictions sur le fold de validation
    y_val_pred = rf_model.predict(X_val_fold)
    
    # Calcul des métriques
    precision = precision_score(y_val_fold, y_val_pred)
    recall = recall_score(y_val_fold, y_val_pred)
    f1 = f1_score(y_val_fold, y_val_pred)
    
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    
    # Ajouter les métriques au dictionnaire
    metrics['precision'].append(precision)
    metrics['recall'].append(recall)
    metrics['f1_score'].append(f1)

# Calculer la moyenne et l'écart-type des métriques
print("\n--- Résultats globaux sur tous les folds ---")
for metric_name, values in metrics.items():
    print(f"{metric_name.capitalize()}: {np.mean(values):.4f} (± {np.std(values):.4f})")
    
    


#--------------- Test sur l'ensemble de test ---------------#

#X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)


# Entraîner le modèle final sur l'ensemble d'entraînement SMOTE (tout le training set)
#rf_model.fit(X_train_smote, y_train_smote)

rf_model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_test_pred = rf_model.predict(X_test)

# Calcul des métriques sur l'ensemble de test
print("\n--- Évaluation sur l'ensemble de test ---")
precision_test = precision_score(y_test, y_test_pred)
recall_test = recall_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred)

# Affichage des métriques sur l'ensemble de test
print(f"Precision: {precision_test:.4f}")
print(f"Recall: {recall_test:.4f}")
print(f"F1-Score: {f1_test:.4f}")

# Si tu veux aussi évaluer l'AUC-ROC sur l'ensemble de test
y_test_pred_proba = rf_model.predict_proba(X_test)[:, 1]
auc_roc_test = roc_auc_score(y_test, y_test_pred_proba)
print(f"AUC-ROC (Test Set): {auc_roc_test:.4f}")





#--------------- Optimisation des hyperparamètres ---------------#
    
print('----------------------------------------------------')
print("Lancement du GridSearchCV")
print('\n')

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Modèle XGBoost
rf_model = xgb.XGBClassifier(random_state=42)

# Paramètres à tester
param_grid = {
    'max_depth': [3, 6, 9],  # Profondeur des arbres
    'learning_rate': [0.01, 0.1, 0.2],  # Taux d'apprentissage
    'n_estimators': [50, 100, 200],  # Nombre d'arbres
    'subsample': [0.8, 1.0],  # Fraction des données utilisées pour chaque arbre
    'colsample_bytree': [0.8, 1.0],  # Fraction des features utilisées par arbre
}

# Initialiser GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, 
                           scoring='f1', verbose=1, n_jobs=-1)

# Entraîner le modèle avec GridSearchCV
grid_search.fit(X_train, y_train)

# Meilleurs paramètres trouvés
print(f"Meilleurs hyperparamètres : {grid_search.best_params_}")

# Meilleur modèle trouvé
best_model = grid_search.best_estimator_

# Prédictions sur le jeu de test avec le meilleur modèle
y_test_pred = best_model.predict(X_test)

# Calcul des métriques sur l'ensemble de test
print("\n--- Évaluation sur l'ensemble de test ---")
precision_test = precision_score(y_test, y_test_pred)
recall_test = recall_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred)

# Affichage des métriques sur l'ensemble de test
print(f"Precision: {precision_test:.4f}")
print(f"Recall: {recall_test:.4f}")
print(f"F1-Score: {f1_test:.4f}")

# AUC-ROC sur l'ensemble de test
y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]
auc_roc_test = roc_auc_score(y_test, y_test_pred_proba)
print(f"AUC-ROC (Test Set): {auc_roc_test:.4f}")




#--------------- Feature Importance avec XGBoost ---------------#


import matplotlib.pyplot as plt

# Récupérer les importances des caractéristiques
feature_importances = best_model.feature_importances_

# Créer un DataFrame pour organiser les résultats
import pandas as pd
import numpy as np

feature_importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)

# Afficher les 10 principales caractéristiques
print(feature_importance_df.head(10))

# Visualisation
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df["Feature"], feature_importance_df["Importance"], color='skyblue')
plt.gca().invert_yaxis()
plt.title("Feature Importance (Best Model)")
plt.xlabel("Importance")
plt.show()



#---------------   mise en production  ---------------#

import joblib
joblib.dump(best_model, 'fraud_detection_model.pkl')


joblib.dump(le, 'label_encoder_sexe_titulaire.pkl')

# Sauvegarde des moyennes
joblib.dump(nom_marchand_means, 'nom_marchand_means.pkl')
joblib.dump(ville_titulaire_means, 'ville_titulaire_means.pkl')
joblib.dump(etat_titulaire_means, 'etat_titulaire_means.pkl')
joblib.dump(emploi_titulaire_means, 'emploi_titulaire_means.pkl')
joblib.dump(code_postal_means, 'code_postal_means.pkl')







