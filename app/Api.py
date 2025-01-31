# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 21:06:25 2025

@author: osahl
"""

from flask import Flask, request, jsonify, send_file, render_template
import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)



# Charger le modèle
model = joblib.load('model/fraud_detection_model.pkl')

# Charger les moyennes de chaque catégorie
nom_marchand_means = joblib.load('model/nom_marchand_means.pkl')
ville_titulaire_means = joblib.load('model/ville_titulaire_means.pkl')
etat_titulaire_means = joblib.load('model/etat_titulaire_means.pkl')
emploi_titulaire_means = joblib.load('model/emploi_titulaire_means.pkl')
code_postal_means = joblib.load('model/code_postal_means.pkl')

#Charger le label encoder 
le= joblib.load('model/label_encoder_sexe_titulaire.pkl')




# Route pour télécharger le fichier CSV et obtenir les prédictions
@app.route('/predict', methods=['POST'])
def predict():
    # Vérifier si un fichier a été envoyé
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # Vérifier si le fichier est bien un CSV
    if file.filename == '' or not file.filename.endswith('.csv'):
        return jsonify({'error': 'Invalid file format. Please upload a CSV file.'}), 400
    
    
    

    # Charger le CSV dans un DataFrame
    df_init = pd.read_csv(file)
    
    
    df = df_init.copy()
    
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
    df.rename(columns=rename_columns, inplace=True)
    
    
    

    # Créer une colonne de flag initialisée à False
    df['flag_aberrant'] = False

    # Vérifier les transactions avec un montant négatif ou nul
    df.loc[df['Montant_transaction'] <= 0, 'flag_aberrant'] = True

    # Vérifier les dates de naissance incohérentes (par exemple, des personnes ayant plus de 120 ans)
    df['Date_naissance_titulaire'] = pd.to_datetime(df['Date_naissance_titulaire'], errors='coerce')
    
    
    df.loc[df['Date_naissance_titulaire'] <= pd.Timestamp.today() - pd.Timedelta(days=120*365), 'flag_aberrant'] = True

    df.loc[(df['Latitude_titulaire'] < -90) | (df['Latitude_titulaire'] > 90), 'flag_aberrant'] = True
    df.loc[(df['Longitude_titulaire'] < -180) | (df['Longitude_titulaire'] > 180), 'flag_aberrant'] = True

    df.loc[(df['Latitude_marchand'] < -90) | (df['Latitude_marchand'] > 90), 'flag_aberrant'] = True
    df.loc[(df['Longitude_marchand'] < -180) | (df['Longitude_marchand'] > 180), 'flag_aberrant'] = True
    
    
    df_init['flag_aberrant'] = df['flag_aberrant']
    
    df_process = preprocess_data(df)  # Fonction que tu devras définir


    # Faire les prédictions
    predictions = model.predict(df_process)

    # Ajouter la colonne des prédictions dans le DataFrame
    df_init['fraud_prediction'] = predictions
    
    
    # Sauvegarder le fichier modifié avec un nom unique
    output_filename = f"predictions_{file.filename}"
    df_init.to_csv(output_filename, index=False)
    
    

    # Envoyer le fichier modifié en réponse
    return send_file(output_filename, as_attachment=True)




def preprocess_data(data):

    
    
    data = data.drop(['flag_aberrant'], axis=1)
    
    
    #---------------   Encodage des variables catégorielles-----------------------#
    

    # Target Encoding pour Nom_marchand
    data['Nom_marchand_encoded'] = data['Nom_marchand'].map(nom_marchand_means)
    data = data.drop(['Nom_marchand'], axis=1)

    data = data.drop(['Prenom_titulaire', 'Nom_titulaire'], axis=1)

    data = pd.get_dummies(data, columns=['Categorie_marchand'], drop_first=True)

    # Convertir les colonnes booléennes en valeurs 0 et 1
    data = data.astype({col: 'int' for col in data.select_dtypes(include=['bool']).columns})




    le = LabelEncoder()
    data['Sexe_titulaire'] = le.fit_transform(data['Sexe_titulaire'])



    data = data.drop(['Adresse_titulaire'], axis=1)

    data['Ville_titulaire_encoded'] = data['Ville_titulaire'].map(ville_titulaire_means)
    data['Etat_titulaire_encoded'] = data['Etat_titulaire'].map(etat_titulaire_means)
    data['Emploi_titulaire_encoded'] = data['Emploi_titulaire'].map(emploi_titulaire_means)

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

    data['Code_postal_target'] = data['Code_postal'].map(code_postal_means)

    # Supprimer les colonnes non utiles
    data = data.drop(columns=['Code_postal'])

    

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
    data = data[columns_order]

    

    return data


@app.route('/')
def home():
    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=False)
