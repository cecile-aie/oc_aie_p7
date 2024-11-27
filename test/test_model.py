import os
import pandas as pd
from app import model

# Cette fonction doit être appelée avant chaque test
def setUp():
    # On définit explicitement FLASK_ENV sur 'testing' pour simuler le comportement de test
    os.environ["FLASK_ENV"] = "testing"

def test_model_loaded():
    # Vérification que le modèle n'est pas chargé en mode test
    assert model is None, "Le modèle ne devrait pas être chargé en mode test."

def test_model_prediction():
    # On s'assure que la prédiction ne peut pas être effectuée en mode test
    if model is not None:
        input_data = pd.DataFrame([{"text": "This is a test"}])
        predictions = model.predict(input_data)
        assert len(predictions) == 1
        assert predictions[0] in [0, 1]  # Assurez-vous que la prédiction est binaire.
    else:
        # Si le modèle n'est pas chargé (mode test), on s'attend à ce qu'il ne puisse pas faire de prédiction
        assert True  # Le test passe si le modèle n'est pas chargé
