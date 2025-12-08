# 🚀 Plateforme d'Analyse RH avec Machine Learning

Application web Flask complète pour l'analyse RH avec trois systèmes de prédiction : Attrition, Formation et Segmentation.

## 📋 Fonctionnalités

### 🏠 Page d'Accueil
- Présentation des 3 systèmes de prédiction
- Navigation intuitive vers toutes les sections
- Statistiques en temps réel

### 🔮 Prédiction d'Attrition
- Formulaire complet avec 30+ variables
- Prédiction instantanée avec Random Forest & XGBoost
- Recommandations personnalisées par niveau de risque

### 🎓 Analyse de Besoin de Formation
- Évaluation en 12 critères
- Prédiction basée sur des règles métier
- Actions immédiates prioritisées

### 👥 Segmentation des Employés
- Classification en 4 segments :
  - **Besoin élevé en formation** : Formation intensive requise
  - **Performants moyens** : Développement ciblé
  - **À développer** : Potentiel de croissance élevé
  - **Top talents** : Fast-track et rétention prioritaire
- Recommandations personnalisées par segment
- Actions immédiates adaptées

## 🛠️ Technologies Utilisées

- **Backend**: Flask 3.0.0
- **ML Models**: scikit-learn 1.3.2, XGBoost 2.0.3
- **Data Processing**: pandas 2.1.4, numpy 1.26.2, scipy 1.11.4
- **Visualisation**: matplotlib 3.8.2, seaborn 0.13.0, plotly 5.18.0
- **Serialization**: joblib 1.3.2
- **Containerisation**: Docker & Docker Compose

## 📊 Modèles de Machine Learning

### 1. Attrition (Random Forest & XGBoost)
- Précision: ~85-87%
- Variables: 30+ features
- Modèles stockés dans `PickleFiles/`

### 2. Formation (Règles métier)
- Système basé sur scoring
- 7 critères pondérés
- Seuil de décision: score >= 4

### 3. Segmentation (KMeans + Random Forest)
- 4 clusters d'employés
- Preprocessing avec ColumnTransformer
- PCA pour réduction dimensionnelle
- **Modèles requis** : `preprocessor.joblib`, `pca.joblib`, `kmeans.joblib`, `classifier.joblib`

## 🔧 Installation et Lancement

### Prérequis pour la Segmentation

Les modèles de segmentation doivent être entraînés avant utilisation. Si vous avez le fichier `train_model.py` dans le dossier `Segmentation/`:

```bash
cd Segmentation
python train_model.py
```

Cela générera les fichiers nécessaires :
- `preprocessor.joblib`
- `pca.joblib`
- `kmeans.joblib`
- `classifier.joblib`
- `feature_names.joblib`

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
- `GET /prediction` - Formulaire de prédiction d'attrition
- `GET /formation` - Formulaire d'analyse de besoin de formation
- `GET /segmentation` - Formulaire de segmentation des employés
- `POST /predict` - Effectuer une prédiction d'attrition
- `POST /predict_formation` - Effectuer une prédiction de besoin de formation
- `POST /predict_segmentation` - Effectuer une segmentation
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
