import pytest
from flask import Flask
from app import app  # Assurez-vous que l'application Flask est importée correctement

# chemin vers le répertoire parent
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page_get(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Entrez une phrase (Tweet)" in response.data

def test_submit_text_success(client):
    response = client.post("/", data={"text": "Bonjour, ceci est un test"})
    assert response.status_code == 200
    assert b"Phrase soumise :" in response.data
    assert "Sentiment prédit :".encode("utf-8") in response.data

def test_submit_text_error(client):
    long_text = "A" * 501  # Texte dépassant la limite de 500 caractères
    response = client.post("/", data={"text": long_text})
    assert response.status_code == 200
    assert b"Le texte est trop long" in response.data

def test_feedback_received(client):
    response = client.post("/", data={"text": "Bonjour, ceci est un test", "feedback": "incorrect"})
    assert response.status_code == 200
    assert b"Merci pour votre retour !" in response.data