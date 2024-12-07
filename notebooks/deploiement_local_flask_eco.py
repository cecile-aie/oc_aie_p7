import pandas as pd
from flask import Flask, request, render_template, jsonify
import mlflow.pyfunc
import mlflow
from deep_translator import GoogleTranslator  # Bibliothèque pour la traduction

# URI pour l'instance MLflow locale
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Nom du modèle et alias
model_name = "Pipeline_sia_lr"  # Nom du modèle enregistré dans le model registry
model_alias = "champion_eco"  # Alias pour accéder à la bonne version du modèle

# Charger le modèle depuis MLflow avec l'alias spécifique
model = mlflow.pyfunc.load_model(f"models:/{model_name}@{model_alias}")

# Initialiser Flask
app = Flask(__name__)

# Initialiser le traducteur
# translator = Translator()

# Fonction pour traduire le texte en anglais
def translate_to_english(text):
    return GoogleTranslator(source='auto', target='en').translate(text)

# Page d'accueil avec un formulaire pour soumettre des phrases
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        text_input = request.form.get("text")
        if text_input:
            try:
                # Traduire le texte en anglais
                translated_text = translate_to_english(text_input)
                
                # Effectuer la prédiction avec le modèle
                input_data = pd.DataFrame([{"text": translated_text}])
                predictions = model.predict(input_data)
                sentiment = "Positif" if predictions[0] == 0 else "Négatif"
                
                return render_template(
                    "index.html", 
                    sentiment=sentiment, 
                    input_text=text_input, 
                    translated_text=translated_text
                )
            except Exception as e:
                return render_template("index.html", error=f"Erreur : {str(e)}")
        else:
            return render_template("index.html", error="Veuillez entrer une phrase.")
    return render_template("index.html")

# Définir l'endpoint pour faire des prédictions via API
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        original_text = data.get("text", "")
        
        # Traduire le texte en anglais
        translated_text = translate_to_english(original_text)
        
        # Effectuer la prédiction
        input_data = pd.DataFrame([{"text": translated_text}])
        predictions = model.predict(input_data)
        
        return jsonify({
            "original_text": original_text,
            "translated_text": translated_text,
            "prediction": predictions.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001)
