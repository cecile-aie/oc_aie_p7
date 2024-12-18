# Streamlit (basique)
import streamlit as st
import mlflow.pyfunc
import pandas as pd

# URI du serveur MLFlow
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")


# Chargement du modèle à partir du Model Registry avec l'alias "champion"
model_name = "BertModelSequenceClassificationFromPretrained"
loaded_model = mlflow.pyfunc.load_model(f"models:/{model_name}@champion")

st.title("Analyse de Sentiment avec BERT")

# Champ de saisie de texte pour l'utilisateur
sentence = st.text_input("Entrez un texte à analyser")

if st.button("Analyser"):
    if sentence:
        # Préparation de l'entrée pour le modèle
        data = pd.DataFrame({"text": [sentence]})
        
        # Prédiction du modèle
        prediction = loaded_model.predict(data)
        
        # Affichage du résultat
        if prediction[0] == 0:
            st.success("Sentiment : Positif 😊")
        else:
            st.error("Sentiment : Négatif 😠")
    else:
        st.warning("Veuillez entrer un texte à analyser.")
