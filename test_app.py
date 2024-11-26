import pytest
from app import app  # Assurez-vous que l'application Flask est importée correctement

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_get(client):
    response = client.get("/")
    print(response.data)  # Affiche le contenu HTML renvoyé
    assert response.status_code == 200