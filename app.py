import os
import pandas as pd
from flask import Flask, request, render_template, jsonify, send_from_directory, render_template_string
import mlflow.pyfunc
from deep_translator import GoogleTranslator
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
from datetime import datetime

# Configuration OpenTelemetry pour Azure Application Insights
resource = Resource.create(attributes={"service.name": "Analyse de Sentiment de Tweets"})
trace.set_tracer_provider(TracerProvider(resource=resource))
instrumentation_key = "4071129e-e96b-494a-b0c7-24e6dac41b18"
exporter = AzureMonitorTraceExporter.from_connection_string(
    f"InstrumentationKey={instrumentation_key}"
)
span_processor = SimpleSpanProcessor(exporter)
trace.get_tracer_provider().add_span_processor(span_processor)
tracer = trace.get_tracer(__name__)

# Vérifiez l'état de FLASK_ENV
FLASK_ENV = os.getenv("FLASK_ENV", "production")
print(f"FLASK_ENV actuel : {FLASK_ENV}")

# Détection du mode test
IS_TESTING = FLASK_ENV == "testing"
print(f"Mode test activé : {IS_TESTING}")

# Chemin vers le stockage monté
LOCAL_MODEL_PATH = "/mnt/azureblob"

# Charger le modèle MLflow depuis le chemin monté
def load_model():
    if IS_TESTING:
        # Simulez un modèle fictif pour les tests
        print("En mode test : modèle factice chargé.")
        return lambda x: ["mocked_prediction"]
    if not os.path.exists(LOCAL_MODEL_PATH):
        raise FileNotFoundError(f"Le chemin {LOCAL_MODEL_PATH} n'existe pas ou le stockage n'est pas monté.")
    return mlflow.pyfunc.load_model(LOCAL_MODEL_PATH)

# Charger le modèle uniquement si ce n'est pas un test
model = None
if not IS_TESTING:
    try:
        print("Chargement du modèle en cours...")
        model = load_model()
    except FileNotFoundError as e:
        print(f"Erreur de chargement du modèle : {e}")
        raise RuntimeError("Le modèle n'a pas pu être chargé. Vérifiez le chemin ou le stockage monté.")
else:
    print("Mode test activé : le modèle ne sera pas chargé.")

# Initialiser Flask
app = Flask(__name__)

def translate_to_english(text):
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception as e:
        log_error("TranslationError", e)
        return text

def log_incorrect_prediction(translated_text, sentiment):
    try:
        with tracer.start_as_current_span("IncorrectPrediction") as span:
            span.set_attribute("translated_text", translated_text)
            span.set_attribute("sentiment", sentiment)
            span.set_attribute("timestamp", datetime.utcnow().isoformat())
    except Exception as e:
        print(f"Erreur lors du logging des prédictions incorrectes : {e}")

def log_error(span_name, error):
    with tracer.start_as_current_span(span_name) as span:
        span.set_attribute("error", str(error))

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        text_input = request.form.get("text")
        feedback = request.form.get("feedback")
        MAX_TEXT_LENGTH = 500  

        if text_input and len(text_input) > MAX_TEXT_LENGTH:
            log_error("TextTooLong", f"Texte de {len(text_input)} caractères soumis.")
            return render_template(
                "index.html",
                error="Le texte est trop long. Veuillez entrer un texte de moins de 500 caractères.",
                sentiment=None,
                input_text=None,
                translated_text=None,
                feedback_received=False
            )

        if text_input:
            try:
                translated_text = translate_to_english(text_input)
                input_data = pd.DataFrame([{"text": translated_text}])
                try:
                    predictions = model.predict(input_data)
                except Exception as e:
                    log_error("PredictionError", e)
                    return render_template("index.html", error="Erreur lors de la prédiction.")
                
                sentiment = "Positif" if predictions[0] == 0 else "Négatif"
                if feedback == "incorrect":
                    log_incorrect_prediction(translated_text, sentiment)
                    return render_template(
                        "index.html",
                        sentiment=None,
                        input_text=None,
                        translated_text=None,
                        feedback_received=True,
                    )
                
                return render_template(
                    "index.html", 
                    sentiment=sentiment, 
                    input_text=text_input, 
                    translated_text=translated_text,
                    feedback_received=True,
                )
            except Exception as e:
                log_error("Error", e)
                return render_template("index.html", error=f"Erreur : {str(e)}")
        else:
            return render_template("index.html", error="Veuillez entrer une phrase.")
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extraction des données JSON
        data = request.get_json()
        original_text = data.get("text")
        if not original_text:
            return jsonify({"error": "Texte manquant"}), 400
        
        # Traduction
        try:
            translated_text = translate_to_english(original_text)
        except Exception as e:
            log_error("TranslationError", f"Erreur lors de la traduction : {str(e)}")
            return jsonify({"error": "Erreur lors de la traduction"}), 500
        
        # Prédiction
        input_data = pd.DataFrame([{"text": translated_text}])
        try:
            predictions = model.predict(input_data)
        except Exception as e:
            log_error("PredictionError", f"Erreur lors de la prédiction : {str(e)}")
            return jsonify({"error": "Erreur lors de la prédiction"}), 500
        
        # Retourner la réponse
        predictions_list = predictions.tolist() if hasattr(predictions, 'tolist') else predictions
        return jsonify({
            "original_text": original_text,
            "translated_text": translated_text,
            "prediction": predictions_list
        })
    except Exception as e:
        log_error("Error", f"Erreur générale : {str(e)}")
        return jsonify({"error": "Erreur lors du traitement de la requête"}), 500

# Exposition de la documentation de l'API
@app.route('/openapi.json', methods=['GET'])
def get_openapi_json():
    return send_from_directory('static/api_docs', 'openapi.json')

# Route pour afficher Swagger UI avec CDN (Content Delivery Network - pas de ressource Swagger locale)
@app.route('/docs', methods=['GET'])
def swagger_ui():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Swagger UI</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/4.15.5/swagger-ui.css">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/4.15.5/swagger-ui-bundle.js"></script>
    </head>
    <body>
        <div id="swagger-ui"></div>
        <script>
            window.onload = function() {
                SwaggerUIBundle({
                    url: '/openapi.json',
                    dom_id: '#swagger-ui',
                });
            };
        </script>
    </body>
    </html>
    """
    return render_template_string(html_content)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
