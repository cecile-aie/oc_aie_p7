import pandas as pd
from flask import Flask, request, render_template, jsonify
import mlflow.pyfunc
import mlflow

# URI pour l'instance MLflow locale
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Nom du modèle et alias
model_name = "BertModelSequenceClassificationFromPretrained"  # Nom du modèle enregistré dans le model registry
model_alias = "champion"  # Alias pour accéder à la bonne version du modèle

# Charge le modèle depuis MLflow avec l'alias spécifique
model = mlflow.pyfunc.load_model(f"models:/{model_name}@{model_alias}")

# Initialiser Flask
app = Flask(__name__)

# Page d'accueil avec un formulaire pour soumettre des phrases
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        text_input = request.form.get("text")
        if text_input:
            # Effectuer la prédiction avec le modèle
            input_data = pd.DataFrame([{"text": text_input}])
            predictions = model.predict(input_data)
            sentiment = "Positif" if predictions[0] == 0 else "Négatif"
            return render_template("index.html", sentiment=sentiment, input_text=text_input)
        else:
            return render_template("index.html", error="Veuillez entrer une phrase.")
    return render_template("index.html")

# Définir l'endpoint pour faire des prédictions via API
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_data = pd.DataFrame([data])
        predictions = model.predict(input_data)
        return jsonify({"prediction": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001)
