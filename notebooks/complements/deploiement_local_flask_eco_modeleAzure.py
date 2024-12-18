import pandas as pd
from flask import Flask, request, render_template, jsonify
import os
from azure.storage.blob import BlobServiceClient
import mlflow.pyfunc

# Paramètres Azure
AZURE_CONNECTION_STRING = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')  
AZURE_CONTAINER_NAME = "oc-p7-ecomodele"  # Nom du conteneur Azure

# Chemin local temporaire pour le modèle
LOCAL_MODEL_PATH = "./azure_model"

# Télécharger tous les fichiers du conteneur Azure Blob Storage
def download_model_from_azure():
    if not os.path.exists(LOCAL_MODEL_PATH):
        os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)

        # Connexion à Azure Blob
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)

        # Lister et télécharger tous les fichiers du conteneur
        blobs = container_client.list_blobs()
        for blob in blobs:
            blob_client = container_client.get_blob_client(blob.name)
            local_file_path = os.path.join(LOCAL_MODEL_PATH, blob.name)

            # Créer les sous-dossiers si nécessaire
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            # Télécharger le fichier
            with open(local_file_path, "wb") as f:
                f.write(blob_client.download_blob().readall())
            print(f"Téléchargé : {blob.name}")
    
    return LOCAL_MODEL_PATH

# Charger le modèle MLflow
def load_model():
    model_path = download_model_from_azure()
    
    # Vérification de l'existence des artefacts requis
    model_meta_path = os.path.join(model_path, "MLmodel")
    if not os.path.exists(model_meta_path):
        raise FileNotFoundError(f"Fichier requis manquant : {model_meta_path}")
    
    # Chargement du modèle via MLflow
    return mlflow.pyfunc.load_model(model_path)

# Initialiser le modèle
try:
    model = load_model()
    print("Modèle chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    model = None

# Initialiser Flask
app = Flask(__name__)

# Page d'accueil avec un formulaire pour soumettre des phrases
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        text_input = request.form.get("text")
        if text_input:
            try:
                # Effectuer la prédiction avec le modèle
                input_data = pd.DataFrame([{"text": text_input}])
                predictions = model.predict(input_data)
                sentiment = "Positif" if predictions[0] == 0 else "Négatif"
                return render_template("index.html", sentiment=sentiment, input_text=text_input)
            except Exception as e:
                return render_template("index.html", error=f"Erreur lors de la prédiction : {e}")
        else:
            return render_template("index.html", error="Veuillez entrer une phrase.")
    return render_template("index.html")

# Définir l'endpoint pour faire des prédictions via API
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Modèle non chargé"}), 500

    try:
        data = request.get_json()
        input_data = pd.DataFrame([data])
        predictions = model.predict(input_data)
        return jsonify({"prediction": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001)
