import os
import pytest
from azure.storage.blob import BlobServiceClient

# Paramètres Azure
AZURE_CONNECTION_STRING = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')  
AZURE_CONTAINER_NAME = "oc-p7-ecomodele"  # Nom du conteneur Azure
REQUIRED_FILES = ["MLmodel", "conda.yaml", "model.pkl"]  # Fichiers nécessaires

@pytest.fixture
def azure_blob_client():
    """
    Fixture pour se connecter au conteneur Azure Blob Storage.
    """
    if not AZURE_CONNECTION_STRING:
        pytest.fail("AZURE_STORAGE_CONNECTION_STRING n'est pas défini.")
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)
        return container_client
    except Exception as e:
        pytest.fail(f"Erreur de connexion au Blob Storage : {str(e)}")

def test_files_in_blob_container(azure_blob_client):
    """
    Teste si les fichiers requis pour le modèle sont présents dans le conteneur Azure Blob Storage.
    """
    missing_files = []
    try:
        # Lister les blobs disponibles
        blobs = azure_blob_client.list_blobs()
        blob_names = {blob.name for blob in blobs}

        # Vérifier chaque fichier requis
        for required_file in REQUIRED_FILES:
            if required_file not in blob_names:
                missing_files.append(required_file)

        # Assertions
        assert not missing_files, f"Fichiers manquants : {', '.join(missing_files)}"

    except Exception as e:
        pytest.fail(f"Erreur lors de la vérification des fichiers dans le conteneur : {str(e)}")
