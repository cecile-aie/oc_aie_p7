import pytest
from unittest.mock import patch, MagicMock
from app import app  # Assurez-vous que 'app' est bien importé de votre application

# Fixture pour configurer l'application Flask en mode test
@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

# Fixture pour patcher le modèle avec un mock
@pytest.fixture
def mock_model():
    with patch('app.model') as mock:
        # Créez un mock pour la méthode `predict`
        mock_predict = MagicMock()
        mock_predict.return_value = [[0.1, 0.9]]  # Exemple de prédiction, peut être ajusté
        mock.predict = mock_predict
        yield mock

# Test de la route /predict avec un mock de modèle
@pytest.mark.parametrize("text_input, expected_status_code", [
    ("Hello", 200),  # Un texte simple qui devrait être traduit et prédit
    ("", 400),  # Texte vide, devrait renvoyer une erreur
])
def test_predict(client, mock_model, text_input, expected_status_code):
    # Simulez une requête POST vers /predict
    response = client.post("/predict", json={"text": text_input})

    # Affichez la réponse pour déboguer
    print("Response status code:", response.status_code)
    print("Response data:", response.data)

    # Vérifiez que le code de statut de la réponse est celui attendu
    assert response.status_code == expected_status_code

    # Si le texte n'est pas vide, vérifiez que la prédiction a été effectuée
    if text_input:
        mock_model.predict.assert_called_once()
        assert "prediction" in response.json
