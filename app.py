import os
import pandas as pd
import numpy as np
import joblib
import json
import plotly
import plotly.express as px
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Load Models and Data
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(MODEL_DIR, 'data_FINAL_with_segments.csv')

def load_models():
    try:
        preprocessor = joblib.load(os.path.join(MODEL_DIR, 'preprocessor.joblib'))
        pca = joblib.load(os.path.join(MODEL_DIR, 'pca.joblib'))
        kmeans = joblib.load(os.path.join(MODEL_DIR, 'kmeans.joblib'))
        classifier = joblib.load(os.path.join(MODEL_DIR, 'classifier.joblib'))
        feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.joblib'))
        return preprocessor, pca, kmeans, classifier, feature_names
    except FileNotFoundError:
        return None, None, None, None, None

preprocessor, pca, kmeans, classifier, feature_names = load_models()

# Segment Names Mapping
SEGMENT_NAMES = {
    0: "Besoin élevé en formation",
    1: "Performants moyens",
    2: "À développer",
    3: "Top talents"
}

@app.route('/')
def dashboard():
    # Load data for dashboard
    if not os.path.exists(DATA_PATH):
        return "Data file not found. Please ensure 'data_FINAL_with_segments.csv' is in the directory."
    
    df = pd.read_csv(DATA_PATH, delimiter=';')
    
    # --- KPIs ---
    total_employees = len(df)
    avg_score = df['avg_training_score'].mean()
    top_talents_count = df[df['segment_name'] == 'Top talents'].shape[0]
    top_talents_pct = (top_talents_count / total_employees) * 100
    
    # --- Visualizations ---
    
    # 1. Segment Distribution (Donut Chart)
    fig_pie = px.pie(df, names='segment_name', title='Répartition des Segments', hole=0.4,
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_pie.update_layout(showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    graph_pie = json.dumps(fig_pie, cls=plotly.utils.PlotlyJSONEncoder)
    
    # 2. Avg Training Score by Segment (Violin Plot for better density view)
    fig_box = px.violin(df, x='segment_name', y='avg_training_score', color='segment_name', 
                     title='Distribution des Scores de Formation', box=True, points="all")
    fig_box.update_layout(showlegend=False)
    graph_box = json.dumps(fig_box, cls=plotly.utils.PlotlyJSONEncoder)
    
    # 3. Previous Year Rating vs Avg Training Score
    fig_scatter = px.scatter(df, x='previous_year_rating', y='avg_training_score', color='segment_name',
                             title='Performance vs Formation', opacity=0.7,
                             size='length_of_service', hover_data=['department'])
    graph_scatter = json.dumps(fig_scatter, cls=plotly.utils.PlotlyJSONEncoder)

    # 4. Feature Importance (Bar Chart)
    if classifier and feature_names:
        importances = classifier.feature_importances_
        # Create DataFrame for plotting
        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        fi_df = fi_df.sort_values(by='Importance', ascending=False).head(10) # Top 10
        
        fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h', 
                        title='Top 10 Facteurs Influents (Random Forest)',
                        color='Importance', color_continuous_scale='Viridis')
        fig_fi.update_layout(yaxis={'categoryorder':'total ascending'})
        graph_fi = json.dumps(fig_fi, cls=plotly.utils.PlotlyJSONEncoder)
    else:
        graph_fi = None

    return render_template('dashboard.html', 
                           graph_pie=graph_pie, 
                           graph_box=graph_box, 
                           graph_scatter=graph_scatter,
                           graph_fi=graph_fi,
                           total_employees=total_employees,
                           avg_score=round(avg_score, 1),
                           top_talents_pct=round(top_talents_pct, 1))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if preprocessor is None:
            return "Models not loaded. Please run 'train_model.py' first."

        # Get form data
        data = {
            'department': request.form['department'],
            'region': request.form['region'],
            'education': request.form['education'],
            'gender': request.form['gender'],
            'recruitment_channel': request.form['recruitment_channel'],
            'no_of_trainings': int(request.form['no_of_trainings']),
            'age': int(request.form['age']),
            'previous_year_rating': float(request.form['previous_year_rating']),
            'length_of_service': int(request.form['length_of_service']),
            'KPIs_met_gt_80': int(request.form['KPIs_met_gt_80']),
            'awards_won?': int(request.form['awards_won']),
            'avg_training_score': float(request.form['avg_training_score'])
        }
        
        # Create DataFrame
        input_df = pd.DataFrame([data])
        
        # Feature Engineering
        # Normalisation du rating (1-5 -> 20-100)
        input_df['previous_year_rating_norm'] = input_df['previous_year_rating'] * 20
        
        # Écart entre score de formation et performance passée
        # Note: In notebook, fillna was used. Here we assume input is complete or handle it.
        # If previous_year_rating is missing (user input), we might need to handle it. 
        # But form requires it.
        input_df['performance_gap'] = input_df['avg_training_score'] - input_df['previous_year_rating_norm']
        
        # Preprocess
        try:
            X_processed = preprocessor.transform(input_df)
            
            # Use Classifier if available, else fallback to KMeans
            if classifier:
                segment = classifier.predict(X_processed)[0]
            else:
                X_pca = pca.transform(X_processed)
                segment = kmeans.predict(X_pca)[0]
                
            segment_name = SEGMENT_NAMES.get(segment, "Unknown")
            
            # Recommendation Logic
            recommendations = {
                "Besoin élevé en formation": "Prioritize intensive training programs and mentorship. Monitor progress closely.",
                "Performants moyens": "Encourage skill diversification. Identify specific areas for improvement to boost performance.",
                "À développer": "Provide targeted training to bridge the gap. Focus on soft skills and leadership potential.",
                "Top talents": "Fast-track for promotion. Assign challenging projects and leadership roles to retain talent."
            }
            recommendation = recommendations.get(segment_name, "No specific recommendation.")
            
            return render_template('result.html', segment=segment, segment_name=segment_name, data=data, recommendation=recommendation)
        except Exception as e:
            return f"Error during prediction: {str(e)}"

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
