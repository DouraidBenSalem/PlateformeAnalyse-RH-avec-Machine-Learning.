# 🚀 Système de Prédiction d'Attrition des Employés

Application web Flask complète pour prédire le risque d'attrition des employés avec recommandations RH personnalisées, utilisant Random Forest et XGBoost.

## 📋 Fonctionnalités

### 🏠 Page d'Accueil
- Présentation du système et de ses capacités
- Navigation intuitive vers toutes les sections
- Statistiques du modèle en temps réel

### 🔮 Prédiction d'Attrition
- Formulaire complet avec 30+ variables
- Prédiction instantanée avec deux modèles (Random Forest & XGBoost)
- Affichage des probabilités de rétention/attrition
- Recommandations personnalisées par niveau de risque

### 📚 Guide des Recommandations RH
- Stratégies pour employés à risque élevé
- Meilleures pratiques pour employés stables
- Indicateurs clés à surveiller
- Actions immédiates prioritisées

## 🛠️ Technologies Utilisées

- **Backend**: Flask 3.0.0
- **ML Models**: scikit-learn 1.3.2, XGBoost 2.0.3
- **Data Processing**: pandas 2.1.4, numpy 1.26.2, scipy 1.11.4
- **Visualisation**: matplotlib 3.8.2, seaborn 0.13.0
- **Containerisation**: Docker & Docker Compose

## 📊 Modèles de Machine Learning

### Random Forest
- 100 arbres de décision
- Optimisé avec GridSearchCV
- Précision: ~85%
- Meilleur pour l'interprétabilité

### XGBoost
- Gradient Boosting optimisé
- Hyperparamètres tunés avec RandomizedSearchCV
- Précision: ~87%
- Meilleur pour la performance

## 🔧 Installation et Lancement

### Méthode 1: Docker (Recommandée)

```bash
# Cloner le repository
git clone <repository-url>
cd Prjt_ML

# Lancer avec Docker Compose
docker-compose up --build

# Accéder à l'application
# http://localhost:5000
```

### Méthode 2: Installation Locale

```bash
# Installer les dépendances
pip install -r requirements.txt

# S'assurer que les modèles sont dans PickleFiles/
# - PickleFiles/rf_model.pkl
# - PickleFiles/xgb_model.pkl
# - PickleFiles/feature_columns.pkl

# Lancer l'application
python app.py

# Accéder à l'application
# http://localhost:5000
```

## 📁 Structure du Projet

```
Prjt_ML/
├── app.py                                # Application Flask principale
├── Employee.ipynb                        # Notebook d'entraînement des modèles
├── Recommandation/
│   └── data_report_final.py             # Script de préparation des données
├── PickleFiles/                         # Modèles entraînés
│   ├── rf_model.pkl
│   ├── xgb_model.pkl
│   ├── feature_columns.pkl
│   └── feature_columns_recommandation.pkl
├── templates/
│   ├── home.html                        # Page d'accueil
│   ├── index.html                       # Formulaire de prédiction
│   ├── result.html                      # Résultats et recommandations
│   └── recommendations.html             # Guide RH complet
├── requirements.txt                     # Dépendances Python
├── Dockerfile                           # Configuration Docker
├── docker-compose.yml                   # Orchestration Docker
└── README.md                            # Documentation

```

## 🎯 Utilisation

### 1. Accéder à la Page d'Accueil
Naviguez vers `http://localhost:5000` pour voir la présentation du système.

### 2. Faire une Prédiction
1. Cliquez sur "Lancer une Prédiction"
2. Remplissez le formulaire avec les informations de l'employé
3. Cliquez sur "Prédire le Risque d'Attrition"
4. Consultez les résultats et recommandations

### 3. Consulter le Guide RH
Accédez aux recommandations générales via "Voir les Recommandations" pour des stratégies RH complètes.

## 📊 Variables Analysées

### Démographiques
- Âge, Genre, État Civil, Distance Domicile-Travail

### Professionnelles
- Département, Poste, Niveau, Années dans l'entreprise

### Formation & Expérience
- Niveau d'éducation, Domaine, Expérience totale

### Rémunération
- Revenu mensuel, Augmentation salariale, Actions

### Conditions de Travail
- Heures supplémentaires, Voyages, Équilibre vie-travail

### Satisfaction
- Environnement, Travail, Relations, Implication

## 🎨 Types de Recommandations

### Risque Élevé (Prediction = 1)
- ⚠️ Intervention urgente dans les 48h
- 💰 Révision salariale prioritaire
- 🎯 Plan de carrière personnalisé
- 🎓 Formation intensive
- ⏰ Amélioration équilibre vie-travail

### Risque Faible (Prediction = 0)
- 🌟 Programme de reconnaissance
- 🚀 Projets stimulants
- 🎓 Formation continue (optionnelle)
- ⚖️ Maintien de la flexibilité
- 📊 Suivi régulier

## 🔍 API Endpoints

- `GET /` - Page d'accueil
- `GET /prediction` - Formulaire de prédiction
- `GET /recommendations` - Guide des recommandations
- `POST /predict` - Effectuer une prédiction
- `GET /api/health` - Vérification de l'état de l'application

## ⚠️ Prérequis

- Python 3.11+
- Docker & Docker Compose (pour déploiement conteneurisé)
- Modèles entraînés dans `PickleFiles/`

## 🐛 Dépannage

**Erreur "Modèle non disponible"**
- Vérifiez que les fichiers `.pkl` existent dans `PickleFiles/`
- Exécutez le notebook `Employee.ipynb` pour générer les modèles

**Erreur de prédiction**
- Assurez-vous que toutes les valeurs du formulaire sont correctes
- Vérifiez la compatibilité des colonnes avec le modèle

**Problème Docker**
- Vérifiez que Docker est en cours d'exécution
- Assurez-vous que le port 5000 est disponible

## 📈 Métriques de Performance

| Modèle | Accuracy | F1-Score | ROC-AUC |
|--------|----------|----------|---------|
| Random Forest | ~85% | ~0.82 | ~0.88 |
| XGBoost | ~87% | ~0.84 | ~0.90 |

## 👥 Contributeurs

Projet académique - 4BI4 ML

## 📝 Licence

Projet académique

---

**Note**: Ce système est un outil d'aide à la décision. Les recommandations doivent être évaluées dans le contexte spécifique de chaque entreprise et employé.
