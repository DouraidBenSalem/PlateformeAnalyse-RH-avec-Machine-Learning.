# Utiliser une image Python officielle
FROM python:3.11-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers requirements
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier tous les fichiers du projet
COPY . .

# Exposer le port sur lequel Flask s'exécute
EXPOSE 5000

# Variable d'environnement pour Flask
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Commande pour démarrer l'application
CMD ["python", "app.py"]
