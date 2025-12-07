# 🚀 Déploiement de Prédiction d'Attrition avec Flask

Application web Flask pour prédire le risque d'attrition des employés en utilisant un modèle Random Forest optimisé.

## 📋 Prérequis

- Python 3.8+
- pip

## 🔧 Installation

1. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

## 📊 Préparation du Modèle

Avant de lancer l'application, vous devez d'abord entraîner et exporter le modèle :

1. Ouvrez le notebook `Employee.ipynb`
2. Exécutez toutes les cellules jusqu'à la fin
3. La dernière cellule va créer deux fichiers :
   - `rf_model.pkl` : Le modèle Random Forest optimisé
   - `feature_columns.pkl` : La liste des colonnes de features

## 🚀 Lancement de l'Application

```bash
python app.py
```

L'application sera accessible à l'adresse : `http://localhost:5000`

## 📁 Structure du Projet

```
ML/
├── app.py                          # Application Flask principale
├── Employee.ipynb                  # Notebook d'entraînement du modèle
├── requirements.txt                # Dépendances Python
├── rf_model.pkl                    # Modèle entraîné (généré par le notebook)
├── feature_columns.pkl             # Colonnes de features (généré par le notebook)
├── templates/
│   ├── index.html                  # Page de formulaire
│   └── result.html                 # Page de résultats
└── HR_*.csv                        # Fichiers de données
```

## 🎯 Utilisation

1. Accédez à `http://localhost:5000`
2. Remplissez le formulaire avec les informations de l'employé
3. Cliquez sur "Prédire le Risque d'Attrition"
4. Consultez les résultats et recommandations

## 📊 Features Utilisées

Le modèle utilise 30+ features incluant :
- **Démographiques** : Âge, Genre, État Civil, Distance Domicile-Travail
- **Professionnelles** : Département, Poste, Niveau, Années dans l'entreprise
- **Formation** : Niveau d'éducation, Domaine, Expérience totale
- **Rémunération** : Revenu mensuel, Augmentation salariale, Actions
- **Conditions** : Heures supplémentaires, Voyages, Équilibre vie-travail
- **Satisfaction** : Environnement, Travail, Relations, Implication

## 🎯 Modèle

- **Algorithme** : Random Forest (100 arbres)
- **Optimisation** : GridSearchCV avec validation croisée
- **Métriques** : ROC-AUC, Accuracy, Precision, Recall

## ⚠️ Notes Importantes

- Assurez-vous que les fichiers `.pkl` existent avant de lancer l'application
- Si les fichiers ne sont pas présents, exécutez d'abord le notebook `Employee.ipynb`
- Le modèle nécessite toutes les features dans le bon ordre

## 🐛 Dépannage

**Erreur "Modèle non disponible"** :
- Exécutez le notebook `Employee.ipynb` complètement
- Vérifiez que `rf_model.pkl` et `feature_columns.pkl` existent

**Erreur de prédiction** :
- Vérifiez que toutes les valeurs du formulaire sont correctes
- Assurez-vous que les colonnes correspondent au modèle entraîné

## 📝 Licence

Projet académique - 4BI4 ML
