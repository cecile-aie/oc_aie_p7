import pandas as pd
from app import model

def test_model_loaded():
    assert model is not None

def test_model_prediction():
    input_data = pd.DataFrame([{"text": "This is a test"}])
    predictions = model.predict(input_data)
    assert len(predictions) == 1
    assert predictions[0] in [0, 1]  # Assurez-vous que la prédiction est binaire.