from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import warnings
import os
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Variables globales pour le modèle
model = None
feature_columns = None
preprocessing_info = None

# Chargement du modèle et des informations de prétraitement
try:
    if os.path.exists('rf_model.pkl'):
        with open('rf_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✅ Modèle chargé depuis rf_model.pkl")
    
    if os.path.exists('feature_columns.pkl'):
        with open('feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        print("✅ Colonnes chargées depuis feature_columns.pkl")
    
    if os.path.exists('preprocessing_info.pkl'):
        with open('preprocessing_info.pkl', 'rb') as f:
            preprocessing_info = pickle.load(f)
        print("✅ Informations de prétraitement chargées")
    
    if model is None:
        print("⚠️ Aucun modèle trouvé. Veuillez exécuter le notebook Employee.ipynb pour entraîner le modèle.")
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
    """Génère une recommandation personnalisée basée sur la prédiction"""
    if prediction == 1:
        recommendations = []
        recommendations.append("⚠️ Cet employé présente un RISQUE ÉLEVÉ d'attrition.")
        
        # Recommandations basées sur les données
        if data.get('OverTime', 'No') == 'Yes':
            recommendations.append("• Réduire les heures supplémentaires et améliorer l'équilibre vie-travail")
        
        if data.get('MonthlyIncome', 5000) < 5000:
            recommendations.append("• Envisager une augmentation de salaire")
        
        if data.get('YearsAtCompany', 0) < 2:
            recommendations.append("• Renforcer l'intégration et le mentorat pour les nouveaux employés")
        
        if data.get('EnvironmentSatisfaction', 3) < 3:
            recommendations.append("• Améliorer l'environnement de travail")
        
        if data.get('JobSatisfaction', 3) < 3:
            recommendations.append("• Organiser un entretien pour comprendre les sources d'insatisfaction")
        
        recommendations.append("• Planifier un entretien individuel rapidement")
        recommendations.append("• Proposer des opportunités de développement de carrière")
        
        return "\n".join(recommendations)
    else:
        return "✅ Cet employé présente un FAIBLE RISQUE d'attrition. Continuez à maintenir un bon environnement de travail."


@app.route('/')
def home():
    """Page d'accueil avec le formulaire de prédiction"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Effectue la prédiction d'attrition"""
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
        
        # Générer la recommandation
        recommendation = generate_recommendation(prediction, display_data)
        
        # Préparer le résultat
        result = {
            'prediction': int(prediction),
            'prediction_label': "Risque d'Attrition Élevé ⚠️" if prediction == 1 else "Risque d'Attrition Faible ✅",
            'probability_stay': f"{prediction_proba[0]*100:.2f}%",
            'probability_leave': f"{prediction_proba[1]*100:.2f}%",
            'recommendation': recommendation,
            'input_data': display_data
        }
        
        return render_template('result.html', result=result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": f"Erreur lors de la prédiction: {str(e)}"
        }), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
