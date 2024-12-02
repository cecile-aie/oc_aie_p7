![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/cecile-aie/oc_aie_p7)


# Projet: Détectez les Bad Buzz grâce au Deep Learning

Réalisé dans le cadre du projet P7 OpenClassRooms Artificial Intelligence Engineer - Réalisez une analyse de sentiments <br>

Il s'agut de préparer un prototype fonctionnel du modèle. Le modèle est exposé via une API déployée sur le Cloud, appelée par une interface locale qui envoie un tweet à l’API et récupère la prédiction de sentiment pour réaliser un feedback.

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
- modèles avancés: embeddings utilisant transformers, la classification est réalisée par des réseaux de neurones sur mesure. L'utilisation de Bert (pré-entrainé, transfer learning ou fine-tuning) est incluse.  

La démarche et le travail réalisé sont détaillés dans l'article [p7_ArticleBlog](blog/p7_ArticleBlog.md)

# Utilisation de l'API

Interface web: [Analyse de sentiment des tweets](https://tweetsentimentanalysiseco-fuetaqf3hbezegch.francecentral-01.azurewebsites.net/)

Documentation de l'API <br>
[UI de la documentation](https://tweetsentimentanalysiseco-fuetaqf3hbezegch.francecentral-01.azurewebsites.net/docs)
<br>
[Accès direct à la documentation en json](https://tweetsentimentanalysiseco-fuetaqf3hbezegch.francecentral-01.azurewebsites.net/openapi.json)  <br>

# Environnement

Utiliser [requirements](requirements.txt), Python 3.11 recommandé.