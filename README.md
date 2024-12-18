![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/cecile-aie/oc_aie_p7)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/cecile-aie/oc_aie_p7)


# Projet: Détectez les Bad Buzz grâce au Deep Learning

Réalisé dans le cadre du projet P7 OpenClassRooms Artificial Intelligence Engineer - Réalisez une analyse de sentiments <br>

Il s'agit de préparer un prototype fonctionnel d'un modèle d'analyse de sentiments. Le modèle est exposé via une API déployée sur le Cloud, appelée par une interface locale qui envoie un tweet à l’API et récupère la prédiction de sentiment pour réaliser un feedback.

## Objectifs:

- Mettre en œuvre un logiciel de version de code
- Suivre la performance d’un modèle en production et en assurer la maintenance
- Concevoir ou ré-utiliser des modèles d'apprentissage profond pré-entraînés
- Concevoir un déploiement continu d'un moteur d’inférence sur une plateforme Cloud
- Définir et mettre en œuvre un pipeline d’entraînement des modèles
- Évaluer la performance des modèles d’apprentissage profond

## Approche:
Elaboration et comparaison de modèles de complexité croissante.<br>
- approche classique: embedding simple suivi d'un classifieur binaire (comprend également l'utilisation d'API sur étagère)
- modèles avancés: embeddings utilisant transformers, la classification est réalisée par des réseaux de neurones sur mesure. L'utilisation de Bert (pré-entrainé, transfer learning ou fine-tuning) est explorée, mais aussi des modèles plus anciens comme Word2Vec, Glove et USE.  

La démarche et le travail réalisé sont détaillés dans l'article [Mise en oeuvre de MLOPS lors de l'élaboration d'un modèle d'analyse de sentiment](blog/p7_ArticleBlog.md)

# Utilisation de l'API

Interface web: [Analyse de sentiment des tweets](https://tweetseco-aqb3breuc4f6bsaj.francecentral-01.azurewebsites.net/)

Documentation de l'API <br>
[UI de la documentation](https://tweetseco-aqb3breuc4f6bsaj.francecentral-01.azurewebsites.net/docs)
<br>
[Accès direct à la documentation en json](https://tweetseco-aqb3breuc4f6bsaj.francecentral-01.azurewebsites.net/openapi.json)  <br>

# Réutilisation du code

## Données

Un jeu de données [Open Source](https://www.kaggle.com/datasets/kazanova/sentiment140) a été utilisé. Tout échantillon de textes étiquettés avec au minimum les colonnes ["text", "target"] est acceptable.

## Environnement

Utiliser [requirements](requirements.txt), Python 3.9 recommandé.

## Notebooks

[Dossier github notebooks et compléments](https://github.com/cecile-aie/oc_aie_p7/tree/main/notebooks)<br>

Versions html:
- [Prétraitement](./notebooks/P7_preprocessing.html)
- [Modèle classique](./notebooks/P7_approche_classique.html)
- [Modèles avancés](./notebooks/P7_modele_avance.html)

## Script d'exécution de l'application

Le [script](app.py) peut être exécuté en local et inclus dans un conteneur de déploiement. Il comprend les tests unitaires et l'envoi de traces via opentelemetry. 

# Outils et méthodes utilisés

[#Python](https://www.python.org/)<br>
[#NLTK](https://www.nltk.org/) [#Spacy](https://spacy.io/) [#Scikit-learn](https://scikit-learn.org/stable/index.html)<br>
[#Pycaret](https://pycaret.org/) [#auto-sklearn](https://automl.github.io/auto-sklearn/master/) [#Keras tuner](https://keras.io/keras_tuner/) [#Optuna](https://optuna.org/)<br>
[#HuggingFace](https://huggingface.co/) [#Keras](https://keras.io/) [#Pytorch](https://pytorch.org/) [#Tensorflow](https://www.tensorflow.org/?hl=fr)<br>

[#Jupyter](https://jupyter.org/) [#Visual Studio Code](https://code.visualstudio.com/) [#Git](https://git-scm.com/) [#Github](https://github.com/) [#Pytest](https://docs.pytest.org/en/stable/) [#Postman](https://www.postman.com/)<br>

[#Azure](https://portal.azure.com)





