# Employee Segmentation Flask App

This application visualizes employee data and predicts employee segments using a Machine Learning model (PCA + KMeans).

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Train the Model:**
    Run the training script to generate the model files (`.joblib`).
    ```bash
    python train_model.py
    ```

3.  **Run the Application:**
    Start the Flask server.
    ```bash
    python app.py
    ```

4.  **Access the App:**
    Open your browser and go to `http://127.0.0.1:5000`.

## Files

*   `app.py`: Main Flask application.
*   `train_model.py`: Script to train and save the ML models.
*   `templates/`: HTML templates for the web interface.
*   `static/`: CSS styles.
*   `data_FINAL_with_segments.csv`: The dataset used for training and visualization.
