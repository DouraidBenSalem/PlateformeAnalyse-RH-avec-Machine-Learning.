from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import warnings
import os
import math
from scipy import stats
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ============================================================
# VARIABLES GLOBALES
# ============================================================
model_rf = None
model_xgb = None
feature_columns = None
feature_columns_recommandation = None
preprocessing_info = None

# ============================================================
# CHARGEMENT DES MODÈLES
# ============================================================
try:
    # Chemins pour les modèles de prédiction d'attrition
    model_rf_path = 'PickleFiles/rf_model.pkl' if os.path.exists('PickleFiles/rf_model.pkl') else 'rf_model.pkl'
    model_xgb_path = 'PickleFiles/xgb_model.pkl' if os.path.exists('PickleFiles/xgb_model.pkl') else 'xgb_model.pkl'
    features_path = 'PickleFiles/feature_columns.pkl' if os.path.exists('PickleFiles/feature_columns.pkl') else 'feature_columns.pkl'
    features_reco_path = 'PickleFiles/feature_columns_recommandation.pkl' if os.path.exists('PickleFiles/feature_columns_recommandation.pkl') else 'feature_columns_recommandation.pkl'
    
    if os.path.exists(model_rf_path):
        with open(model_rf_path, 'rb') as f:
            model_rf = pickle.load(f)
        print(f"✅ Modèle Random Forest chargé depuis {model_rf_path}")
    
    if os.path.exists(model_xgb_path):
        with open(model_xgb_path, 'rb') as f:
            model_xgb = pickle.load(f)
        print(f"✅ Modèle XGBoost chargé depuis {model_xgb_path}")
    
    if os.path.exists(features_path):
        with open(features_path, 'rb') as f:
            feature_columns = pickle.load(f)
        print(f"✅ Colonnes chargées depuis {features_path}")
    
    if os.path.exists(features_reco_path):
        with open(features_reco_path, 'rb') as f:
            feature_columns_recommandation = pickle.load(f)
        print(f"✅ Colonnes de recommandation chargées depuis {features_reco_path}")
    
    if os.path.exists('preprocessing_info.pkl'):
        with open('preprocessing_info.pkl', 'rb') as f:
            preprocessing_info = pickle.load(f)
        print("✅ Informations de prétraitement chargées")
    
    if model_rf is None and model_xgb is None:
        print("⚠️ Aucun modèle trouvé. Veuillez exécuter le notebook pour entraîner les modèles.")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle: {e}")


def preprocess_input(data):
    """Prétraite les données d'entrée selon le même pipeline que l'entraînement"""
    df = pd.DataFrame([data])
    
    # Encoder les variables catégorielles binaires
    binary_mapping = {
        'OverTime': {'No': 0, 'Yes': 1},
        'Gender': {'Female': 0, 'Male': 1},
        'BusinessTravel': {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2},
        'MaritalStatus': {'Single': 0, 'Married': 1, 'Divorced': 2}
    }
    
    for col, mapping in binary_mapping.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    
    # One-Hot Encoding pour Department, EducationField, JobRole
    nominal_cols = ['Department', 'EducationField', 'JobRole']
    df_encoded = pd.get_dummies(df, columns=nominal_cols, drop_first=False, dtype=int)
    
    # Ajouter les colonnes manquantes avec 0
    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Réorganiser les colonnes dans le bon ordre
    df_encoded = df_encoded[feature_columns]
    
    return df_encoded


def generate_recommendation(prediction, data):
    """Génère une recommandation personnalisée basée sur la prédiction d'attrition"""
    experience = data.get('TotalWorkingYears', 0)
    years_at_company = data.get('YearsAtCompany', 0)
    monthly_income = data.get('MonthlyIncome', 0)
    job_satisfaction = data.get('JobSatisfaction', 3)
    
    if prediction == 1:
        # RISQUE ÉLEVÉ
        recommandations = []
        
        # Recommandations basées sur l'expérience
        if experience < 5:
            recommandations.append({
                "titre": "🎯 Programme de Mentorat Junior",
                "description": "Mettre en place un programme de mentorat avec un senior pour accélérer le développement.",
                "priorite": "Haute"
            })
        elif experience < 10:
            recommandations.append({
                "titre": "📈 Formation en Leadership",
                "description": "Développer les compétences managériales pour une évolution vers des postes de responsabilité.",
                "priorite": "Haute"
            })
        
        # Recommandations basées sur les données
        if data.get('OverTime', 'No') == 'Yes':
            recommandations.append({
                "titre": "⏰ Réduction Heures Supplémentaires",
                "description": "Réduire les heures supplémentaires et améliorer l'équilibre vie-travail.",
                "priorite": "Critique"
            })
        
        if monthly_income < 5000:
            recommandations.append({
                "titre": "💰 Révision Salariale",
                "description": "Envisager une augmentation de salaire pour rester compétitif sur le marché.",
                "priorite": "Critique"
            })
        
        if years_at_company < 2:
            recommandations.append({
                "titre": "🤝 Renforcement Intégration",
                "description": "Renforcer l'intégration et le mentorat pour les nouveaux employés.",
                "priorite": "Haute"
            })
        
        if data.get('EnvironmentSatisfaction', 3) < 3:
            recommandations.append({
                "titre": "🏢 Amélioration Environnement",
                "description": "Améliorer l'environnement de travail (espaces, équipements, ambiance).",
                "priorite": "Haute"
            })
        
        if job_satisfaction < 3:
            recommandations.append({
                "titre": "💬 Entretien Individuel",
                "description": "Organiser un entretien pour comprendre les sources d'insatisfaction.",
                "priorite": "Critique"
            })
        
        recommandations.append({
            "titre": "🎯 Plan de Carrière Personnalisé",
            "description": "Établir un plan d'évolution sur 2-3 ans avec des objectifs clairs.",
            "priorite": "Haute"
        })
        
        return {
            "niveau": "Risque Élevé",
            "couleur": "danger",
            "icon": "exclamation-triangle",
            "titre": "🚨 Intervention Urgente Requise",
            "message": "Cet employé présente un risque élevé d'attrition. Une intervention rapide est nécessaire.",
            "recommandations": recommandations,
            "actions_immediates": [
                "📅 Organiser un entretien individuel dans les 48h",
                "💡 Proposer un plan de développement personnalisé",
                "💰 Réviser le package de rémunération",
                "🎓 Identifier les besoins de formation",
                "👥 Assigner un mentor ou coach interne"
            ]
        }
    else:
        # RISQUE FAIBLE
        recommandations = [
            {
                "titre": "🌟 Programme de Reconnaissance",
                "description": "Mettre en place un système de reconnaissance des contributions exceptionnelles.",
                "priorite": "Moyenne"
            },
            {
                "titre": "🎓 Formation Continue",
                "description": "Proposer des formations de développement personnel et technique.",
                "priorite": "Basse"
            },
            {
                "titre": "🚀 Projets Stimulants",
                "description": "Offrir des opportunités de projets innovants et challengeants.",
                "priorite": "Moyenne"
            },
            {
                "titre": "⚖️ Équilibre Vie Pro/Perso",
                "description": "Maintenir la flexibilité (télétravail, horaires aménagés).",
                "priorite": "Haute"
            }
        ]
        
        return {
            "niveau": "Risque Faible",
            "couleur": "success",
            "icon": "check-circle",
            "titre": "✅ Employé Stable et Engagé",
            "message": "Cet employé montre un bon niveau d'engagement. Continuez les bonnes pratiques.",
            "recommandations": recommandations,
            "actions_immediates": [
                "🎊 Reconnaître publiquement les contributions",
                "📈 Proposer des objectifs stimulants",
                "🤝 Favoriser le partage d'expérience",
                "🎯 Offrir des opportunités de projets transverses",
                "📊 Maintenir un suivi régulier (trimestriel)"
            ]
        }


# ============================================================
# ROUTES DE L'APPLICATION
# ============================================================

@app.route('/')
def home():
    """Page d'accueil principale"""
    return render_template('home.html')


@app.route('/prediction')
def prediction_page():
    """Page de prédiction avec le formulaire"""
    return render_template('index.html')


@app.route('/recommendations')
def recommendations():
    """Page de recommandations RH générales"""
    stats = {
        'total_features': 30,
        'model_accuracy': 85,
        'trees_count': 100,
        'prediction_time': 'Temps réel'
    }
    return render_template('recommendations.html', stats=stats)


@app.route('/formation')
def formation_page():
    """Page de prédiction de besoin de formation"""
    return render_template('indexreco.html')


@app.route('/predict_formation', methods=['POST'])
def predict_formation():
    """Effectue la prédiction du besoin de formation"""
    
    try:
        # Récupération des données du formulaire
        data = {
            'city': request.form.get('city'),
            'city_development_index': float(request.form.get('city_development_index')),
            'gender': request.form.get('gender'),
            'relevent_experience': request.form.get('relevent_experience'),
            'enrolled_university': request.form.get('enrolled_university'),
            'education_level': request.form.get('education_level'),
            'major_discipline': request.form.get('major_discipline'),
            'experience': float(request.form.get('experience')),
            'company_size': float(request.form.get('company_size')),
            'company_type': request.form.get('company_type'),
            'last_new_job': float(request.form.get('last_new_job')),
            'training_hours': int(request.form.get('training_hours'))
        }
        
        # Logique de prédiction basée sur des règles métier
        # (Alternative au modèle ML qui pose problème)
        prediction = predict_formation_rule_based(data)
        
        # Générer la recommandation de formation
        recommendation = generate_formation_recommendation(prediction, data)
        
        # Préparer le résultat
        result = {
            'prediction': int(prediction),
            'prediction_label': "Formation Nécessaire" if prediction == 1 else "Formation Optionnelle",
            'recommendation': recommendation,
            'input_data': data
        }
        
        return render_template('resultatrecom.html', result=result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": f"Erreur lors de la prédiction: {str(e)}"
        }), 400


def predict_formation_rule_based(data):
    """
    Prédiction basée sur des règles métier au lieu du modèle ML
    Retourne 1 si formation nécessaire, 0 sinon
    """
    score = 0
    
    # Critère 1: Heures de formation faibles
    training_hours = data.get('training_hours', 0)
    if training_hours < 30:
        score += 3
    elif training_hours < 50:
        score += 2
    elif training_hours < 80:
        score += 1
    
    # Critère 2: Expérience faible
    experience = data.get('experience', 0)
    if experience < 5:
        score += 2
    elif experience < 10:
        score += 1
    
    # Critère 3: Pas d'expérience pertinente
    if data.get('relevent_experience', '') == 'No relevent experience':
        score += 2
    
    # Critère 4: Pas d'inscription universitaire en cours
    enrolled = data.get('enrolled_university', '')
    if 'no_enrollment' in enrolled.lower():
        score += 1
    
    # Critère 5: Niveau d'éducation faible
    education = data.get('education_level', '')
    if 'High School' in education or 'Primary' in education:
        score += 2
    elif 'Graduate' in education and 'Masters' not in education and 'Phd' not in education:
        score += 1
    
    # Critère 6: Changement de job récent
    last_new_job = data.get('last_new_job', 0)
    if last_new_job <= 1:
        score += 1
    
    # Critère 7: Index de développement de la ville faible
    city_dev = data.get('city_development_index', 1.0)
    if city_dev < 0.7:
        score += 1
    
    # Si le score est >= 4, formation nécessaire
    return 1 if score >= 4 else 0


def generate_formation_recommendation(prediction, input_data):
    """
    Génère des recommandations de formation basées sur la prédiction
    prediction=1 : Formation NÉCESSAIRE (employé risque de partir)
    prediction=0 : Formation OPTIONNELLE (employé stable)
    """
    
    # Analyser le profil pour des recommandations personnalisées
    experience = input_data.get('experience', 0)
    training_hours = input_data.get('training_hours', 0)
    education_level = input_data.get('education_level', '')
    enrolled_university = input_data.get('enrolled_university', '')
    
    if prediction == 1:
        # FORMATION NÉCESSAIRE
        recommandations = []
        
        # Recommandations basées sur l'expérience
        if experience < 5:
            recommandations.append({
                "titre": "🎯 Programme de Mentorat Junior",
                "description": "Mettre en place un programme de mentorat avec un senior pour accélérer le développement des compétences.",
                "priorite": "Haute"
            })
        elif experience < 10:
            recommandations.append({
                "titre": "📈 Formation en Leadership",
                "description": "Développer les compétences managériales pour une évolution vers des postes de responsabilité.",
                "priorite": "Haute"
            })
        else:
            recommandations.append({
                "titre": "🌟 Programme de Spécialisation Expert",
                "description": "Renforcer l'expertise technique avec des formations avancées et des certifications professionnelles.",
                "priorite": "Haute"
            })
        
        # Recommandations basées sur les heures de formation actuelles
        if training_hours < 20:
            recommandations.append({
                "titre": "📚 Formation Technique Intensive",
                "description": "Au moins 40 heures de formation sur les nouvelles technologies et méthodologies du domaine.",
                "priorite": "Critique"
            })
        elif training_hours < 50:
            recommandations.append({
                "titre": "🔧 Ateliers Pratiques Avancés",
                "description": "Compléter par 30 heures supplémentaires de workshops hands-on et projets réels.",
                "priorite": "Haute"
            })
        
        # Recommandations basées sur l'éducation
        if 'no_enrollment' in enrolled_university.lower():
            recommandations.append({
                "titre": "🎓 Programme de Formation Continue",
                "description": "Proposer une prise en charge pour suivre des cours universitaires en ligne (Coursera, edX, Udacity).",
                "priorite": "Moyenne"
            })
        
        recommandations.extend([
            {
                "titre": "💼 Plan de Carrière Personnalisé",
                "description": "Établir un plan d'évolution sur 2-3 ans avec des objectifs clairs et mesurables.",
                "priorite": "Critique"
            },
            {
                "titre": "💰 Révision de la Rémunération",
                "description": "Évaluer et ajuster le package de rémunération pour rester compétitif sur le marché.",
                "priorite": "Haute"
            },
            {
                "titre": "🎯 Projets Stimulants",
                "description": "Assigner des projets challengeants alignés avec les aspirations professionnelles.",
                "priorite": "Moyenne"
            }
        ])
        
        return {
            "formation_necessaire": "OUI - URGENT",
            "formation_couleur": "danger",
            "niveau": "Formation Nécessaire",
            "couleur": "danger",
            "icon": "exclamation-triangle",
            "titre": "🚨 Formation Urgente Requise",
            "message": "L'analyse indique que cet employé nécessite une formation immédiate pour améliorer sa rétention et son engagement.",
            "raison": "Risque élevé de désengagement détecté. Une intervention rapide avec un programme de formation adapté est recommandée.",
            "recommandations": recommandations,
            "actions_immediates": [
                "📅 Organiser un entretien individuel dans les 48h",
                "📋 Évaluer les besoins de formation spécifiques",
                "💡 Proposer un plan de développement personnalisé",
                "🎓 Inscrire à minimum 2 formations prioritaires",
                "👥 Assigner un mentor ou coach interne"
            ]
        }
    
    else:
        # FORMATION OPTIONNELLE
        recommandations = []
        
        # Recommandations de maintien et d'enrichissement
        if experience > 10:
            recommandations.append({
                "titre": "🌟 Veille Technologique",
                "description": "Participer à des conférences et événements du secteur pour rester à jour.",
                "priorite": "Basse"
            })
        
        if training_hours > 50:
            recommandations.append({
                "titre": "👨‍🏫 Programme de Mentorat Inverse",
                "description": "Partager son expertise en formant d'autres employés moins expérimentés.",
                "priorite": "Moyenne"
            })
        
        recommandations.extend([
            {
                "titre": "🎓 Formations de Développement Personnel",
                "description": "Soft skills, gestion du temps, communication avancée (optionnel).",
                "priorite": "Basse"
            },
            {
                "titre": "🚀 Innovation et Créativité",
                "description": "Encourager la participation à des hackathons et projets d'innovation.",
                "priorite": "Moyenne"
            },
            {
                "titre": "🏆 Programme de Reconnaissance",
                "description": "Mettre en place un système de reconnaissance des contributions exceptionnelles.",
                "priorite": "Moyenne"
            },
            {
                "titre": "⚖️ Équilibre Vie Pro/Perso",
                "description": "Maintenir la flexibilité (télétravail, horaires aménagés).",
                "priorite": "Haute"
            }
        ])
        
        return {
            "formation_necessaire": "NON - Optionnelle",
            "formation_couleur": "success",
            "niveau": "Formation Optionnelle",
            "couleur": "success",
            "icon": "check-circle",
            "titre": "✅ Employé Stable et Engagé",
            "message": "L'employé montre un bon niveau d'engagement. La formation reste recommandée pour le développement continu.",
            "raison": "Profil stable et satisfait. Les formations proposées visent à maintenir la motivation et favoriser l'évolution professionnelle.",
            "recommandations": recommandations,
            "actions_immediates": [
                "🎊 Reconnaître publiquement les contributions",
                "📈 Proposer des objectifs stimulants",
                "🤝 Favoriser le partage d'expérience",
                "🎯 Offrir des opportunités de projets transverses",
                "📊 Maintenir un suivi régulier (trimestriel)"
            ]
        }


@app.route('/predict', methods=['POST'])
def predict():
    """Effectue la prédiction d'attrition"""
    # Utiliser le modèle Random Forest par défaut, sinon XGBoost
    model = model_rf if model_rf is not None else model_xgb
    
    if model is None:
        return jsonify({
            "error": "Modèle non disponible. Veuillez entraîner le modèle d'abord."
        }), 500
    
    try:
        # Récupération des données du formulaire (données brutes)
        raw_data = {
            'Age': int(request.form.get('Age')),
            'DailyRate': int(request.form.get('DailyRate')),
            'DistanceFromHome': int(request.form.get('DistanceFromHome')),
            'Education': int(request.form.get('Education')),
            'EmployeeCount': 1,  # Constante
            'EmployeeNumber': 0,  # Placeholder
            'EnvironmentSatisfaction': int(request.form.get('EnvironmentSatisfaction')),
            'HourlyRate': int(request.form.get('HourlyRate')),
            'JobInvolvement': int(request.form.get('JobInvolvement')),
            'JobLevel': int(request.form.get('JobLevel')),
            'JobSatisfaction': int(request.form.get('JobSatisfaction')),
            'MonthlyRate': int(request.form.get('MonthlyRate')),
            'Over18': 0,  # Constante
            'PercentSalaryHike': int(request.form.get('PercentSalaryHike')),
            'RelationshipSatisfaction': int(request.form.get('RelationshipSatisfaction')),
            'StandardHours': 80,  # Constante
            'WorkLifeBalance': int(request.form.get('WorkLifeBalance')),
            'Gender': request.form.get('Gender'),
            'BusinessTravel': request.form.get('BusinessTravel'),
            'Department': request.form.get('Department'),
            'EducationField': request.form.get('EducationField'),
            'JobRole': request.form.get('JobRole'),
            'MaritalStatus': request.form.get('MaritalStatus'),
            'OverTime': request.form.get('OverTime')
        }
        
        # Calculs de transformations (log, sqrt)
        import math
        num_companies = int(request.form.get('NumCompaniesWorked', 1))
        years_in_role = int(request.form.get('YearsInCurrentRole', 0))
        years_with_manager = int(request.form.get('YearsWithCurrManager', 0))
        years_at_company = int(request.form.get('YearsAtCompany', 0))
        monthly_income = float(request.form.get('MonthlyIncome', 1000))
        stock_option = int(request.form.get('StockOptionLevel', 0))
        training_times = int(request.form.get('TrainingTimesLastYear', 0))
        performance_rating = int(request.form.get('PerformanceRating', 3))
        years_since_promo = int(request.form.get('YearsSinceLastPromotion', 0))
        total_working_years = int(request.form.get('TotalWorkingYears', 0))
        
        raw_data['NumCompaniesWorked_log'] = math.log(max(num_companies, 1) + 1)
        raw_data['YearsInCurrentRole_log'] = math.log(max(years_in_role, 0) + 1)
        raw_data['YearsWithCurrManager_log'] = math.log(max(years_with_manager, 0) + 1)
        raw_data['YearsAtCompany_log'] = math.log(max(years_at_company, 0) + 1)
        raw_data['MonthlyIncome_log'] = math.log(max(monthly_income, 1))
        raw_data['StockOptionLevel_log'] = math.log(max(stock_option, 0) + 1)
        raw_data['TrainingTimesLastYear_sqrt'] = math.sqrt(max(training_times, 0))
        raw_data['PerformanceRating_sqrt'] = math.sqrt(max(performance_rating, 0))
        raw_data['YearsSinceLastPromotion_sqrt'] = math.sqrt(max(years_since_promo, 0))
        raw_data['TotalWorkingYears'] = total_working_years
        
        # Prétraiter les données
        input_df = preprocess_input(raw_data)
        
        # Effectuer la prédiction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        # Données pour les recommandations
        display_data = {
            'Age': raw_data['Age'],
            'Gender': request.form.get('Gender'),
            'MaritalStatus': request.form.get('MaritalStatus'),
            'Department': request.form.get('Department'),
            'JobRole': request.form.get('JobRole'),
            'JobLevel': raw_data['JobLevel'],
            'MonthlyIncome': monthly_income,
            'TotalWorkingYears': total_working_years,
            'YearsAtCompany': years_at_company,
            'OverTime': request.form.get('OverTime'),
            'JobSatisfaction': raw_data['JobSatisfaction'],
            'WorkLifeBalance': raw_data['WorkLifeBalance'],
            'EnvironmentSatisfaction': raw_data['EnvironmentSatisfaction']
        }
        
        # Générer les recommandations
        recommendation = generate_recommendation(prediction, display_data)
        
        # Préparer le résultat
        result = {
            'prediction': int(prediction),
            'prediction_label': "Risque d'Attrition Élevé ⚠️" if prediction == 1 else "Risque d'Attrition Faible ✅",
            'probability_stay': f"{prediction_proba[0]*100:.2f}%",
            'probability_leave': f"{prediction_proba[1]*100:.2f}%",
            'recommendation': recommendation,
            'input_data': display_data,
            'model_used': 'Random Forest' if model == model_rf else 'XGBoost'
        }
        
        return render_template('result.html', result=result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": f"Erreur lors de la prédiction: {str(e)}"
        }), 400


@app.route('/api/health')
def health_check():
    """Endpoint de vérification de l'état de l'application"""
    return jsonify({
        "status": "running",
        "models_loaded": {
            "random_forest": model_rf is not None,
            "xgboost": model_xgb is not None
        },
        "features_loaded": feature_columns is not None
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
