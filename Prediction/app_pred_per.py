from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Chargement des objets ML
scaler = joblib.load('scaler.pkl')
pca_model = joblib.load('pca_model.pkl')
model = joblib.load('model_xgb.pkl')


# ----------------- Page d'accueil (formulaire) -----------------
@app.route('/')
def index():
    # valeurs par défaut pour pré-remplir le formulaire
    return render_template(
        'index.html',
        percent_hike='',
        job_satisfaction=3,
        job_involvement=3,
        years_in_role='',
        years_since_last_promotion=''
    )


# ----------------- Prédiction + redirection vers result.html -----------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        percent_hike = float(request.form['percent_salary_hike'])
        job_satisfaction = int(request.form['job_satisfaction'])
        job_involvement = int(request.form['job_involvement'])
        years_in_role = int(request.form['years_in_current_role'])
        years_since_last_promotion = int(request.form['years_since_last_promotion'])
    except (ValueError, KeyError):
        # Retour sur la page du formulaire en cas d’erreur de saisie
        return render_template(
            'index.html',
            error="Veuillez saisir uniquement des nombres valides.",
            percent_hike=request.form.get('percent_salary_hike', ''),
            job_satisfaction=int(request.form.get('job_satisfaction', 3)),
            job_involvement=int(request.form.get('job_involvement', 3)),
            years_in_role=request.form.get('years_in_current_role', ''),
            years_since_last_promotion=request.form.get('years_since_last_promotion', '')
        )

    # ⚠️ DataFrame avec EXACTEMENT les mêmes features que lors de l'entraînement
    X = pd.DataFrame([{
        'PercentSalaryHike': percent_hike,
        'JobSatisfaction': job_satisfaction,
        'JobInvolvement': job_involvement,
        'YearsInCurrentRole_log': np.log1p(years_in_role),
        'YearsSinceLastPromotion_sqrt': np.sqrt(years_since_last_promotion)
    }])

    # Normalisation + PCA
    X_scaled = scaler.transform(X)
    X_pca = pca_model.transform(X_scaled)

    # Prédiction (le modèle XGBoost renvoie 0..3 → on +1 pour avoir 1..4)
    y_pred = model.predict(X_pca)[0]
    prediction = int(y_pred) + 1

    labels = {
        1: "Performance faible",
        2: "Performance moyenne",
        3: "Bonne performance",
        4: "Performance excellente"
    }
    label = labels.get(prediction, "Inconnu")

    # 👉 ICI : on affiche la page result.html
    return render_template(
        'result.html',
        prediction=prediction,
        label=label,
        percent_hike=percent_hike,
        job_satisfaction=job_satisfaction,
        job_involvement=job_involvement,
        years_in_role=years_in_role,
        years_since_last_promotion=years_since_last_promotion
    )


if __name__ == "__main__":
    app.run(debug=True)
