Détection de Fraude - Projet Data Science
Ce projet vise à détecter les paiements frauduleux dans un service financier en utilisant des techniques avancées de data science. Il utilise des méthodes de machine learning pour identifier les transactions suspectes et fournir des prédictions sur la probabilité qu'une transaction soit frauduleuse.

Objectif
L'objectif principal de ce projet est de démontrer l'utilisation de modèles prédictifs pour détecter les fraudes financières dans un environnement bancaire ou financier. Ce modèle peut être utilisé par des institutions financières pour améliorer la sécurité et l'intégrité de leurs services de paiement.

Technologies utilisées
Python
Pandas
Scikit-learn
XGBoost
Matplotlib, Seaborn
Anaconda
Spyder
Description des données
Les données utilisées pour ce projet proviennent d'un ensemble de données de détection de fraude disponible sur Kaggle. Ce jeu de données contient des informations sur des transactions financières, avec une colonne indiquant si chaque transaction est frauduleuse ou non.

Vous pouvez télécharger les données ici : (https://www.kaggle.com/datasets/kartik2112/fraud-detection?)

Installation et exécution
Clonez ce dépôt :

bash
Copier
Modifier
git clone https://github.com/dataAlchemySolutions/Detection_de_fraude.git
Créez un environnement Anaconda et installez les dépendances :

bash
Copier
Modifier
conda create --name fraud-detection-env python=3.8
conda activate fraud-detection-env
pip install -r requirements.txt
Téléchargez les données depuis Kaggle et placez-les dans le répertoire approprié.

Ouvrez le projet dans Spyder et exécutez les scripts Python pour démarrer l'analyse et la modélisation.

Résultats
Ce projet inclut l'entraînement de plusieurs modèles de machine learning, tels que :

Régression logistique
Random Forest
XGBoost
Les modèles sont évalués en utilisant des métriques de performance comme la précision, le rappel et le score F1.

Contribution
Si vous souhaitez contribuer à ce projet, n'hésitez pas à faire une demande de pull (PR). Toute contribution est la bienvenue pour améliorer la détection de fraude !

Auteurs
Data Alchemy - Société de consulting spécialisée en data science
Licence
Ce projet est sous licence MIT - voir le fichier LICENSE pour plus de détails.
