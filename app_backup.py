from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import warnings
import os
import math
import json
import plotly
import plotly.express as px
import joblib
from scipy import stats
from sklearn.preprocessing import LabelEncoder

# Clustering imports
try:
    import umap
    import hdbscan
    import gower_multiprocessing as gower
    CLUSTERING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Clustering dependencies not available: {e}")
    CLUSTERING_AVAILABLE = False

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

# Variables pour la segmentation
preprocessor_seg = None
pca_seg = None
kmeans_seg = None
classifier_seg = None
feature_names_seg = None

# Mapping des noms de segments
SEGMENT_NAMES = {
    0: "Besoin élevé en formation",
    1: "Performants moyens",
    2: "À développer",
    3: "Top talents"
}

# Variables pour le clustering
hdbscan_model = None
umap_reducer = None
scaler_clustering = None
cluster_summary = None
training_data_processed = None
training_labels = None
numerical_cols_clustering = ['length_of_service', 'avg_training_score']
gower_cols = [
    'department', 'education', 'gender', 'recruitment_channel',
    'KPIs_met_more_than_80', 'awards_won', 'trained_more_than_once',
    'length_of_service', 'avg_training_score'
]

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


# ============================================================
# CHARGEMENT DES MODÈLES DE SEGMENTATION
# ============================================================
def load_segmentation_models():
    """Charge les modèles de segmentation"""
    global preprocessor_seg, pca_seg, kmeans_seg, classifier_seg, feature_names_seg
    
    try:
        # Chercher d'abord dans PickleFiles/, sinon dans Segmentation/
        possible_dirs = ['PickleFiles', 'Segmentation', '.']
        
        for seg_dir in possible_dirs:
            preprocessor_path = os.path.join(seg_dir, 'preprocessor.joblib')
            if os.path.exists(preprocessor_path):
                preprocessor_seg = joblib.load(preprocessor_path)
                print(f"✅ Preprocessor de segmentation chargé depuis {preprocessor_path}")
                
                pca_path = os.path.join(seg_dir, 'pca.joblib')
                if os.path.exists(pca_path):
                    pca_seg = joblib.load(pca_path)
                    print(f"✅ PCA de segmentation chargé depuis {pca_path}")
                
                kmeans_path = os.path.join(seg_dir, 'kmeans.joblib')
                if os.path.exists(kmeans_path):
                    kmeans_seg = joblib.load(kmeans_path)
                    print(f"✅ KMeans de segmentation chargé depuis {kmeans_path}")
                
                classifier_path = os.path.join(seg_dir, 'classifier.joblib')
                if os.path.exists(classifier_path):
                    classifier_seg = joblib.load(classifier_path)
                    print(f"✅ Classifier de segmentation chargé depuis {classifier_path}")
                
                feature_names_path = os.path.join(seg_dir, 'feature_names.joblib')
                if os.path.exists(feature_names_path):
                    feature_names_seg = joblib.load(feature_names_path)
                    print(f"✅ Feature names de segmentation chargés depuis {feature_names_path}")
                
                break  # Sortir de la boucle si on a trouvé les fichiers
            
    except Exception as e:
        print(f"❌ Erreur lors du chargement des modèles de segmentation: {e}")

# Charger les modèles de segmentation
load_segmentation_models()


# ============================================================
# CHARGEMENT DES MODÈLES DE CLUSTERING
# ============================================================
def load_clustering_models():
    """Charge les modèles de clustering HDBSCAN"""
    global hdbscan_model, umap_reducer, scaler_clustering, cluster_summary
    global training_data_processed, training_labels
    
    if not CLUSTERING_AVAILABLE:
        print("⚠️ Clustering dependencies not installed. Skipping clustering model loading.")
        return
    
    try:
        clustering_dir = 'PickleFiles/clustering'
        
        if not os.path.exists(clustering_dir):
            print(f"⚠️ Directory {clustering_dir} not found. Clustering models not loaded.")
            return
        
        # Load HDBSCAN model
        hdbscan_path = os.path.join(clustering_dir, 'hdbscan_model.pkl')
        if os.path.exists(hdbscan_path):
            with open(hdbscan_path, 'rb') as f:
                hdbscan_model = pickle.load(f)
            print(f"✅ HDBSCAN model loaded from {hdbscan_path}")
        
        # Load UMAP reducer
        umap_path = os.path.join(clustering_dir, 'umap_reducer.pkl')
        if os.path.exists(umap_path):
            with open(umap_path, 'rb') as f:
                umap_reducer = pickle.load(f)
            print(f"✅ UMAP reducer loaded from {umap_path}")
        
        # Load scaler
        scaler_path = os.path.join(clustering_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler_clustering = pickle.load(f)
            print(f"✅ Scaler loaded from {scaler_path}")
        
        # Load cluster summary
        summary_path = os.path.join(clustering_dir, 'cluster_summary.pkl')
        if os.path.exists(summary_path):
            with open(summary_path, 'rb') as f:
                cluster_summary = pickle.load(f)
            print(f"✅ Cluster summary loaded from {summary_path}")
        
        # Load training data
        training_data_path = os.path.join(clustering_dir, 'training_data_processed.pkl')
        if os.path.exists(training_data_path):
            with open(training_data_path, 'rb') as f:
                training_data_processed = pickle.load(f)
            print(f"✅ Training data loaded from {training_data_path}")
        
        # Load training labels
        labels_path = os.path.join(clustering_dir, 'training_labels.pkl')
        if os.path.exists(labels_path):
            with open(labels_path, 'rb') as f:
                training_labels = pickle.load(f)
            print(f"✅ Training labels loaded from {labels_path}")
        
    except Exception as e:
        print(f"❌ Error loading clustering models: {e}")

# Load clustering models
load_clustering_models()


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
# FONCTION DE PRÉTRAITEMENT POUR CLUSTERING
# ============================================================

def preprocess_clustering_input(data):
    """Preprocesses the input data from the form for clustering."""
    df = pd.DataFrame([data])

    # Feature Engineering from the notebook
    df['trained_more_than_once'] = df['no_of_trainings'].apply(lambda x: 1 if x > 1 else 0)
    
    # Map gender
    df['gender'] = df['gender'].map({'m': 1, 'f': 0})

    # Select and reorder columns to match the training data for Gower
    df = df[gower_cols]

    # Type conversion
    df[numerical_cols_clustering] = df[numerical_cols_clustering].astype(float)
    df[df.columns.difference(numerical_cols_clustering)] = df[df.columns.difference(numerical_cols_clustering)].astype(str)

    # Scaling numerical columns
    if scaler_clustering is not None:
        df[numerical_cols_clustering] = scaler_clustering.transform(df[numerical_cols_clustering])

    return df


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


@app.route('/segmentation')
def segmentation_page():
    """Page de prédiction de segment employé"""
    return render_template('indexseg.html')


@app.route('/formation')
def formation_page():
    """Page de prédiction de besoin de formation"""
    return render_template('indexreco.html')


@app.route('/dashboard')
def dashboard():
    """Dashboard avec visualisations interactives"""
    try:
        # Charger les données
        data_paths = [
            'Datasets/HR_final_after_clean_encode_transform.csv',
            'Datasets/data_FINAL_with_segments.csv',
            'HR_final_after_clean_encode_transform.csv',
            'data_FINAL_with_segments.csv'
        ]
        
        df_hr = None
        df_seg = None
        
        for path in data_paths:
            if os.path.exists(path) and 'HR_final' in path:
                df_hr = pd.read_csv(path)
                print(f"✅ Données HR chargées depuis {path}")
                break
        
        for path in data_paths:
            if os.path.exists(path) and 'segments' in path:
                try:
                    df_seg = pd.read_csv(path, delimiter=';')
                    print(f"✅ Données de segmentation chargées depuis {path}")
                except:
                    df_seg = pd.read_csv(path)
                    print(f"✅ Données de segmentation chargées depuis {path}")
                break
        
        graphs = {}
        stats = {}
        
        # === VISUALISATIONS ATTRITION ===
        if df_hr is not None:
            # Stats générales
            stats['total_employees_hr'] = len(df_hr)
            if 'Attrition' in df_hr.columns:
                stats['attrition_rate'] = round((df_hr['Attrition'].sum() / len(df_hr)) * 100, 1)
                stats['retention_rate'] = round(100 - stats['attrition_rate'], 1)
            
            # 1. Distribution de l'attrition (Pie Chart)
            if 'Attrition' in df_hr.columns:
                attrition_counts = df_hr['Attrition'].value_counts()
                fig_attrition = px.pie(
                    values=attrition_counts.values,
                    names=['Rétention', 'Attrition'],
                    title='Distribution Attrition vs Rétention',
                    hole=0.4,
                    color_discrete_sequence=['#51cf66', '#ff6b6b']
                )
                graphs['attrition_pie'] = json.dumps(fig_attrition, cls=plotly.utils.PlotlyJSONEncoder)
            
            # 2. Attrition par Département (Bar Chart)
            if 'Department' in df_hr.columns and 'Attrition' in df_hr.columns:
                dept_attrition = df_hr.groupby('Department')['Attrition'].agg(['sum', 'count'])
                dept_attrition['rate'] = (dept_attrition['sum'] / dept_attrition['count'] * 100).round(1)
                fig_dept = px.bar(
                    x=dept_attrition.index,
                    y=dept_attrition['rate'],
                    title='Taux d\'Attrition par Département (%)',
                    labels={'x': 'Département', 'y': 'Taux d\'Attrition (%)'},
                    color=dept_attrition['rate'],
                    color_continuous_scale='Reds'
                )
                graphs['dept_bar'] = json.dumps(fig_dept, cls=plotly.utils.PlotlyJSONEncoder)
            
            # 3. Distribution des âges (Histogram)
            if 'Age' in df_hr.columns:
                fig_age = px.histogram(
                    df_hr,
                    x='Age',
                    nbins=20,
                    title='Distribution des Âges',
                    labels={'Age': 'Âge', 'count': 'Nombre d\'employés'},
                    color_discrete_sequence=['#667eea']
                )
                graphs['age_hist'] = json.dumps(fig_age, cls=plotly.utils.PlotlyJSONEncoder)
            
            # 4. Satisfaction vs Attrition (Box Plot)
            if 'JobSatisfaction' in df_hr.columns and 'Attrition' in df_hr.columns:
                fig_satisfaction = px.box(
                    df_hr,
                    x='Attrition',
                    y='JobSatisfaction',
                    title='Satisfaction au Travail vs Attrition',
                    labels={'Attrition': 'Attrition', 'JobSatisfaction': 'Satisfaction'},
                    color='Attrition',
                    color_discrete_map={0: '#51cf66', 1: '#ff6b6b'}
                )
                graphs['satisfaction_box'] = json.dumps(fig_satisfaction, cls=plotly.utils.PlotlyJSONEncoder)
            
            # 5. Revenu vs Attrition (Violin Plot)
            if 'MonthlyIncome' in df_hr.columns and 'Attrition' in df_hr.columns:
                fig_income = px.violin(
                    df_hr,
                    x='Attrition',
                    y='MonthlyIncome',
                    title='Revenu Mensuel vs Attrition',
                    labels={'Attrition': 'Attrition', 'MonthlyIncome': 'Revenu Mensuel'},
                    color='Attrition',
                    box=True,
                    color_discrete_map={0: '#51cf66', 1: '#ff6b6b'}
                )
                graphs['income_violin'] = json.dumps(fig_income, cls=plotly.utils.PlotlyJSONEncoder)
        
        # === VISUALISATIONS SEGMENTATION ===
        if df_seg is not None:
            stats['total_employees_seg'] = len(df_seg)
            
            # 6. Distribution des Segments (Pie Chart)
            if 'segment_name' in df_seg.columns:
                seg_counts = df_seg['segment_name'].value_counts()
                fig_segments = px.pie(
                    values=seg_counts.values,
                    names=seg_counts.index,
                    title='Répartition des Segments RH',
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                graphs['segments_pie'] = json.dumps(fig_segments, cls=plotly.utils.PlotlyJSONEncoder)
            
            # 7. Score de Formation par Segment (Violin Plot)
            if 'segment_name' in df_seg.columns and 'avg_training_score' in df_seg.columns:
                fig_training = px.violin(
                    df_seg,
                    x='segment_name',
                    y='avg_training_score',
                    title='Score de Formation par Segment',
                    color='segment_name',
                    box=True,
                    points='all'
                )
                fig_training.update_layout(showlegend=False)
                graphs['training_violin'] = json.dumps(fig_training, cls=plotly.utils.PlotlyJSONEncoder)
            
            # 8. Performance vs Formation (Scatter)
            if all(col in df_seg.columns for col in ['previous_year_rating', 'avg_training_score', 'segment_name']):
                fig_perf = px.scatter(
                    df_seg,
                    x='previous_year_rating',
                    y='avg_training_score',
                    color='segment_name',
                    title='Performance vs Score de Formation',
                    labels={'previous_year_rating': 'Évaluation Année N-1', 'avg_training_score': 'Score Formation'},
                    size='length_of_service' if 'length_of_service' in df_seg.columns else None,
                    hover_data=['department'] if 'department' in df_seg.columns else None
                )
                graphs['perf_scatter'] = json.dumps(fig_perf, cls=plotly.utils.PlotlyJSONEncoder)
        
        return render_template('dashboard.html', graphs=graphs, stats=stats)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template('error.html', message=f"Erreur lors du chargement du dashboard: {str(e)}")


@app.route('/predict_segmentation', methods=['POST'])
def predict_segmentation():
    """Effectue la prédiction du segment employé"""
    
    try:
        # Récupération des données du formulaire
        data = {
            'department': request.form.get('department'),
            'region': request.form.get('region'),
            'education': request.form.get('education'),
            'gender': request.form.get('gender'),
            'recruitment_channel': request.form.get('recruitment_channel'),
            'no_of_trainings': int(request.form.get('no_of_trainings')),
            'age': int(request.form.get('age')),
            'previous_year_rating': float(request.form.get('previous_year_rating')),
            'length_of_service': int(request.form.get('length_of_service')),
            'KPIs_met_gt_80': int(request.form.get('KPIs_met_gt_80')),
            'awards_won?': int(request.form.get('awards_won')),
            'avg_training_score': float(request.form.get('avg_training_score'))
        }
        
        # Si les modèles ML sont disponibles, les utiliser
        if preprocessor_seg is not None and classifier_seg is not None:
            print("🔮 Utilisation des modèles ML pour la prédiction de segmentation")
            
            # Créer DataFrame
            input_df = pd.DataFrame([data])
            
            # Feature Engineering (comme dans train_model.py)
            input_df['previous_year_rating_norm'] = input_df['previous_year_rating'] * 20
            input_df['performance_gap'] = input_df['avg_training_score'] - input_df['previous_year_rating_norm']
            
            # Prétraitement avec le preprocessor
            X_processed = preprocessor_seg.transform(input_df)
            
            # Prédiction avec le classifier Random Forest
            segment = classifier_seg.predict(X_processed)[0]
            
            print(f"✅ Segment prédit par ML: {segment} ({SEGMENT_NAMES.get(segment)})")
        else:
            print("⚠️ Modèles ML non disponibles, utilisation des règles métier")
            # Sinon, utiliser un système basé sur des règles métier
            segment = predict_segment_rule_based(data)
            print(f"✅ Segment prédit par règles: {segment} ({SEGMENT_NAMES.get(segment)})")
        
        segment_name = SEGMENT_NAMES.get(segment, "Unknown")
        
        # Générer les recommandations
        recommendation = generate_segmentation_recommendation(segment, segment_name, data)
        
        result = {
            'segment': int(segment),
            'segment_name': segment_name,
            'recommendation': recommendation,
            'data': data
        }
        
        return render_template('resultatseg.html', result=result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": f"Erreur lors de la prédiction de segmentation: {str(e)}"
        }), 400


def predict_segment_rule_based(data):
    """
    Prédiction basée sur des règles métier
    Retourne 0-3 correspondant aux segments :
    0: Besoin élevé en formation
    1: Performants moyens
    2: À développer
    3: Top talents
    """
    score = 0
    
    # Critères de performance
    trainings = data.get('no_of_trainings', 0)
    rating = data.get('previous_year_rating', 0)
    avg_score = data.get('avg_training_score', 0)
    kpis_met = data.get('KPIs_met_gt_80', 0)
    awards = data.get('awards_won?', 0)
    
    # Calcul du score de performance
    # Formations (max 3 points)
    if trainings >= 5:
        score += 3
    elif trainings >= 3:
        score += 2
    elif trainings >= 1:
        score += 1
    
    # Rating année précédente (max 4 points)
    if rating >= 4.5:
        score += 4
    elif rating >= 4.0:
        score += 3
    elif rating >= 3.0:
        score += 2
    elif rating >= 2.0:
        score += 1
    
    # Score de formation (max 3 points)
    if avg_score >= 80:
        score += 3
    elif avg_score >= 70:
        score += 2
    elif avg_score >= 60:
        score += 1
    
    # KPIs atteints (2 points)
    if kpis_met == 1:
        score += 2
    
    # Prix remportés (3 points)
    if awards == 1:
        score += 3
    
    # Détermination du segment basé sur le score total (max 15 points)
    if score >= 12:
        return 3  # Top talents
    elif score >= 8:
        return 2  # À développer
    elif score >= 4:
        return 1  # Performants moyens
    else:
        return 0  # Besoin élevé en formation


def generate_segmentation_recommendation(segment, segment_name, data):
    """Génère des recommandations personnalisées basées sur le segment"""
    
    age = data.get('age', 30)
    experience = data.get('length_of_service', 5)
    trainings = data.get('no_of_trainings', 1)
    avg_score = data.get('avg_training_score', 50)
    rating = data.get('previous_year_rating', 3.0)
    
    if segment == 0:  # Besoin élevé en formation
        recommandations = []
        
        if trainings < 3:
            recommandations.append({
                "titre": "📚 Programme de Formation Intensive",
                "description": "Inscrire immédiatement à minimum 3 formations ciblées sur les compétences clés du poste.",
                "priorite": "Critique"
            })
        
        if avg_score < 60:
            recommandations.append({
                "titre": "🎯 Coaching Individuel",
                "description": "Assigner un coach dédié pour améliorer les compétences et la performance.",
                "priorite": "Haute"
            })
        
        if rating < 3.0:
            recommandations.append({
                "titre": "📊 Plan d'Amélioration de Performance",
                "description": "Établir un plan de 90 jours avec des objectifs mesurables et un suivi hebdomadaire.",
                "priorite": "Critique"
            })
        
        recommandations.extend([
            {
                "titre": "👥 Mentorat Professionnel",
                "description": "Associer avec un senior pour accélérer le développement des compétences.",
                "priorite": "Haute"
            },
            {
                "titre": "🔍 Évaluation des Compétences",
                "description": "Réaliser une évaluation complète pour identifier les lacunes spécifiques.",
                "priorite": "Haute"
            }
        ])
        
        return {
            "niveau": "Besoin Élevé en Formation",
            "couleur": "danger",
            "icon": "exclamation-triangle",
            "titre": "🚨 Formation Urgente Requise",
            "message": "Cet employé nécessite une attention immédiate avec un programme de formation intensif.",
            "recommandations": recommandations,
            "actions_immediates": [
                "📅 Organiser un entretien de développement dans les 48h",
                "📋 Créer un plan de formation personnalisé sur 3 mois",
                "👨‍🏫 Assigner un mentor expérimenté",
                "📊 Établir des objectifs SMART mesurables",
                "🔄 Programmer des points de suivi bi-hebdomadaires"
            ]
        }
    
    elif segment == 1:  # Performants moyens
        recommandations = [
            {
                "titre": "📈 Développement Ciblé",
                "description": "Identifier 2-3 compétences clés à développer pour passer au niveau supérieur.",
                "priorite": "Moyenne"
            },
            {
                "titre": "🎓 Formations Spécialisées",
                "description": "Proposer des formations avancées dans le domaine d'expertise.",
                "priorite": "Moyenne"
            },
            {
                "titre": "🤝 Projets Transverses",
                "description": "Impliquer dans des projets inter-départements pour élargir les compétences.",
                "priorite": "Basse"
            },
            {
                "titre": "🎯 Objectifs Ambitieux",
                "description": "Fixer des objectifs challengeants pour stimuler la progression.",
                "priorite": "Moyenne"
            }
        ]
        
        return {
            "niveau": "Performant Moyen",
            "couleur": "warning",
            "icon": "chart-line",
            "titre": "📊 Potentiel d'Amélioration Identifié",
            "message": "Employé avec de bonnes bases. Accompagnement ciblé pour atteindre l'excellence.",
            "recommandations": recommandations,
            "actions_immediates": [
                "🎯 Définir 3 objectifs de développement",
                "📚 Proposer 2 formations spécialisées",
                "🤝 Assigner un projet stimulant",
                "📊 Faire un bilan trimestriel",
                "💡 Encourager l'initiative personnelle"
            ]
        }
    
    elif segment == 2:  # À développer
        recommandations = [
            {
                "titre": "🌱 Programme de Développement",
                "description": "Plan structuré sur 6 mois pour développer les soft skills et compétences techniques.",
                "priorite": "Haute"
            },
            {
                "titre": "💼 Leadership & Communication",
                "description": "Formations en communication, gestion d'équipe et intelligence émotionnelle.",
                "priorite": "Haute"
            },
            {
                "titre": "🎯 Projets à Responsabilité",
                "description": "Confier progressivement des responsabilités pour développer l'autonomie.",
                "priorite": "Moyenne"
            },
            {
                "titre": "📊 Suivi Régulier",
                "description": "Points mensuels pour mesurer les progrès et ajuster le plan.",
                "priorite": "Haute"
            }
        ]
        
        return {
            "niveau": "À Développer",
            "couleur": "info",
            "icon": "seedling",
            "titre": "🌱 Potentiel de Croissance Élevé",
            "message": "Employé avec un fort potentiel. Investissement en développement recommandé.",
            "recommandations": recommandations,
            "actions_immediates": [
                "📋 Créer un plan de développement sur 6 mois",
                "🎓 Inscrire à 2-3 formations soft skills",
                "👥 Organiser des sessions de mentorat",
                "🎯 Fixer des objectifs de progression clairs",
                "📊 Faire un bilan mensuel des progrès"
            ]
        }
    
    else:  # Top talents (segment 3)
        recommandations = [
            {
                "titre": "🚀 Fast-Track Promotion",
                "description": "Préparer activement pour une promotion vers un poste de leadership.",
                "priorite": "Haute"
            },
            {
                "titre": "🎯 Projets Stratégiques",
                "description": "Confier des projets critiques et innovants alignés avec la stratégie d'entreprise.",
                "priorite": "Haute"
            },
            {
                "titre": "👨‍🏫 Devenir Mentor",
                "description": "Encourager à partager l'expertise en mentorant d'autres employés.",
                "priorite": "Moyenne"
            },
            {
                "titre": "💰 Package de Rétention",
                "description": "Proposer un package compétitif (augmentation, bonus, avantages) pour retenir le talent.",
                "priorite": "Critique"
            },
            {
                "titre": "🌟 Reconnaissance Publique",
                "description": "Célébrer les réussites et reconnaître publiquement les contributions.",
                "priorite": "Haute"
            }
        ]
        
        return {
            "niveau": "Top Talent",
            "couleur": "success",
            "icon": "trophy",
            "titre": "🏆 Talent Exceptionnel Identifié",
            "message": "Employé hautement performant. Priorité absolue: rétention et développement.",
            "recommandations": recommandations,
            "actions_immediates": [
                "💰 Réviser immédiatement le package de rémunération",
                "🚀 Établir un plan de carrière sur 2-3 ans",
                "🎯 Confier un projet stratégique majeur",
                "👥 Proposer un rôle de mentor/leader",
                "🌟 Organiser une reconnaissance formelle",
                "📊 Entretien trimestriel avec la direction"
            ]
        }


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


# ============================================================
# ROUTES CLUSTERING
# ============================================================

@app.route('/clustering')
def clustering_index():
    """Page de clustering des employés"""
    return render_template('clustering_index.html')


@app.route('/clustering/predict', methods=['POST'])
def clustering_predict():
    """Prédiction de clustering pour un employé"""
    try:
        # Get data from form
        form_data = {
            'department': request.form['department'],
            'education': request.form['education'],
            'gender': request.form['gender'],
            'recruitment_channel': request.form['recruitment_channel'],
            'no_of_trainings': int(request.form['no_of_trainings']),
            'length_of_service': float(request.form['length_of_service']),
            'KPIs_met_more_than_80': int(request.form['KPIs_met_more_than_80']),
            'awards_won': int(request.form['awards_won']),
            'avg_training_score': float(request.form['avg_training_score'])
        }

        # Check if clustering is available
        if not CLUSTERING_AVAILABLE:
            return render_template('clustering_result.html',
                                 prediction=-1,
                                 summary={"message": "Clustering dependencies not installed. Please install umap-learn, hdbscan, and gower_multiprocessing."},
                                 employee_data=form_data)
        
        # Check if models are loaded
        if training_data_processed is None or training_labels is None:
            return render_template('clustering_result.html',
                                 prediction=-1,
                                 summary={"message": "Clustering models not loaded. Please train the models first."},
                                 employee_data=form_data)

        # Preprocess the input data
        new_employee_processed = preprocess_clustering_input(form_data.copy())

        try:
            # The UMAP model was trained on a precomputed Gower distance matrix.
            # Since UMAP with precomputed distances doesn't support transform(),
            # we'll use a nearest-neighbor approach: find the k closest training points
            # and assign the new employee to the most common cluster among them.
            
            # Identify which columns are categorical for the gower matrix
            cat_features_mask = [col not in numerical_cols_clustering for col in training_data_processed.columns]

            # Calculate Gower distances from the new point to all points in the training set
            gower_dists = gower.gower_matrix(
                new_employee_processed.to_numpy(), 
                training_data_processed.to_numpy(), 
                cat_features=cat_features_mask
            )

            # Find the k nearest neighbors (using k=5 as a reasonable default)
            k = 5
            nearest_indices = np.argsort(gower_dists[0])[:k]
            
            # Get the clusters of the nearest neighbors
            nearest_clusters = training_labels[nearest_indices]
            
            # Assign to the most common cluster among the neighbors (excluding noise -1)
            clusters_without_noise = nearest_clusters[nearest_clusters != -1]
            if len(clusters_without_noise) > 0:
                predicted_cluster = np.bincount(clusters_without_noise).argmax()
            else:
                predicted_cluster = -1  # All neighbors are noise

        except Exception as e:
            # Fallback for demonstration if prediction fails
            print(f"Prediction failed, using fallback. Error: {e}")
            predicted_cluster = -1  # Assign to noise if prediction fails

        # Get cluster summary
        if cluster_summary is not None and predicted_cluster in cluster_summary.index:
            summary = cluster_summary.loc[predicted_cluster].to_dict()
        else:
            summary = {"message": "Cet employé est considéré comme atypique et n'appartient pas à un cluster spécifique."}

        return render_template('clustering_result.html',
                             prediction=predicted_cluster,
                             summary=summary,
                             employee_data=form_data)
    
    except Exception as e:
        print(f"Error in clustering prediction: {e}")
        return render_template('clustering_result.html',
                             prediction=-1,
                             summary={"message": f"Erreur lors de la prédiction: {str(e)}"},
                             employee_data={})


@app.route('/api/health')
def health_check():
    """Endpoint de vérification de l'état de l'application"""
    return jsonify({
        "status": "running",
        "models_loaded": {
            "random_forest": model_rf is not None,
            "xgboost": model_xgb is not None,
            "clustering": training_data_processed is not None
        },
        "features_loaded": feature_columns is not None
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
