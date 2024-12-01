---
title: "Mise en oeuvre de MLOPS dans un cas d'étude d'analyse de sentiments de Tweets"
date: "2024-11-28"
tags: ["Language modelling", "GitHub", "VS Code", "MLFlow"]
categories: ["Partage d'expérience"]
author: "Cécile"
---

# La mission et le contexte

## Le sujet : Analyse de sentiments

Un client dans le domain du transport aérien nous a demandé un prototype fonctionnel d'un modèle permettant de détecter les tweets à connotation négative. Cette problématique liée à la modélisation du langage (Natural Language Processing ou NLP) est déjà largement étudiée. On trouve pour y répondre un arsenal de bibliothèques, de modèles spécifiquement entrainés, voire de services entièrement packagés.<br>
J'ai testé différentes approches avec un objectif focalisé autant sur l'exactitude de prédiction (l'accuracy) que sur le temps d'entrainement et le temps de réponse du modèle une fois déployé.<br>
Ce cas d'étude est également l'occasion de donner un exemple concret de la mise en application de MLOPS


## Les outils : Bibliothèques d'analyse, méthodes de modélisation du langage

Afin de pouvoir mener un calcul de classification il nous faut transformer le texte en chiffres, ou plutôt en vecteurs.
De façon intuitive on imagine qu'en associant une connotation plus ou moins négative à chaque mot on peut arriver à donner un score à une phrase.<br>
Les méthodes utilisées avant le machine learning, basées sur des dictionnaires associant chaque mot à un score sont encore utilisées. J'ai pu tester SentimentIntensityAnalysis (NLTK) et TextBlob.<br>
De façon plus élaborée on peut représenter chaque mot unique (token) par un vecteur dont les composantes sont ses occurences dans les différents tweets (comptage simple) ou encore le rapport entre sa fréquence dans un tweet et celle dans l'ensemble des tweets de l'échantillon (méthode TFIdF). Les modélisation utilisées sont CountVectorizer et TFIdF.<br>
Viennent ensuite des méthodes plus élaborées nécessitant la mise en oeuvre de réseaux de neurones: pour réaliser la modélisation des mots on va considérer leur contexte (les mots précédents et suivants), selon une certaine fenêtre et certaines conditions d'apparition, et les modèles les plus récents permettent de donner plus d'importance à certaines associations de mots (mécanisme d'attention). J'ai exploré successivement Word2Vec, GloVE, USE, Bert et une variation Roberta spécialisée dans l'analyse de tweet. <br>

## La méhtode : MLOPS

![alt text](image.png)

A l'instar de DeOps, MLOps est le trait d'union entre les développeurs et l'opérationnel, intégrant en plus la boucle de Machine Learning. Les étapes sont clairement définies mais quel est l'enjeu et qu'est-ce que cela implique?

### Principes de MLOPS

1 - Automatisation
Au départ, le processus de mise en œuvre d'un modèle est manuel et itératif, incluant la préparation, validation des données, et la création de modèles.<br>
Une fois automatisé, le modèle se forme et se recycle de manière continue, en validant les nouvelles données dès leur disponibilité.<br>
L'automatisation du pipeline CI/CD permet d'intégrer et de déployer des modèles ML de manière continue et sans intervention manuelle.

2 - Intégration continue
L'intégration continue permet de valider les tests, les données, les schémas et les modèles, tout en déployant automatiquement des pipelines ML ou en annulant les modifications non désirées.

3 - Reproductibilité
Stockage de la conception, du traitement des données, de la formation du moèdle, du déploiement afin que les modèles soient facilement reproduits.

### Avantages de MLOPS

- L'automatisation des processus permet le déploiement rapide d'un grand nombre de modèles
- Productivité améliorée grâce à la collaboration et à la réutilisation des modèles
- Les modèles non déployés peuvent être valorisés 
- Les modèles peuvent être surveillés et actualisés 
- Plus de réussite dans les projets grâce à l'intégration, au déploiement, à la livraison, la surveillance et les tests continus des modèles

### Outils choisis
- Pipeline de données: Automatisé dans un notebook<br>

![alt text](image-7.png)
![alt text](image-9.png)
![alt text](image-8.png)


<i> Interface MLFLOW: Expérimentations, métriques et artéfacts loggés </i>

- Pipeline ML: MLFlow utilisé à la fois pour l'enregistrement des expérimentation et des résultats et pour le registre de modèles.<br>
- Pipeline d'application: Avec un dossier de travail configuré comme dépôt local Git, Visual Studio Code possède l'ensemble des extensions permettant de visualiser les modifications du code et de gérer le versionning, puis dans les étapes de développement de réaliser les tests.<br>

# Étape 1 : Analyse et préparation des données

N'ayant pas de données client j'ai utilisé un jeu de données [Open Source](https://www.kaggle.com/datasets/kazanova/sentiment140) contenant 1 600 000 tweets étiquetés postif/négatif de façon équilibrée. 

## Sélection du jeu de données

Afin de couvrir le vocabulaire métier les tweets utilisant les termes typiques du transport aérien on été sélectionnés. J'ai ensuite utilisé LanguageDectector (Spacy) pour limiter aux tweets en langue anglaise.Après ré-équiilibrage par élimination j'ai un jeu de données de 6 473 tweets.

## Nettoyage

Il n'y a pas de valeurs manquantes par contre des doublons ont été détectés. J'ai supprimé ceux pouvant créer une confusion lors de la modélisaton: par exemple ceux ayant le même contenu textuel mais qui ont été jugés soit positif soit négatif. Il y a également les messages identiques multiples d'une même auteur (un seul a été préservé) qui peuvent biaiser les modélisations par comptage de mots.

## Pre-traitement

Cette partie est particulièrement utile pour les méthodes dont le temps de calcul et la taille des matrices résultantes dépendent de la taille du vocabulaire, typiquement les méthodes classiques par comptage. <br>
Les méthodes plus récentes possède des outils de prétraitement, surtout la tokenisation - réductions à des mots uniques. Il est utile de réaliser ces étapes manuellement afin de pouvoir dimensionner par exemple la taille à choisir pour l'homogénéisation des tenseurs en entrée des méthodes les plus élaborées comme WordToVec ou Bert (le padding) ou encore dimensionner la taille maximale des séquences en entrée des réseaux de neurones. 

### Traitement du "langage tweet" 

Les tweets constituent une variante du langage commun avec des expressions exacerbées (répétitions), imagées (écomticon) et l'utilisation de hashtags, d'url, de citations. Pour chaque étape il faut juger si le texte concerné peut avoir une valeur informative. Voici ce qui a été appliqué:

- Détection des expressions héritées de html générées lors du passage en texte brut (ex: &Amp)
- Remplacement des url et des citations par des balises <url> et <mention> <br>
        Original: was totally crushed when I found that much looked forward to plane read: Air Kisses by @zotheysay, had sold out at airport <br>
        <span style="color: green;">Modifié: was totally crushed when I found that much looked forward to plane read: Air Kisses by <mention>, had sold out at airport</span>
        
- Réduction de la répétition des caractères, dans cet exemple les points d'exclamation<br>
        Original: Sitting in the airport, waiting for the plane to arrive, so we can depart!!!   http://twitpic.com/6ebzo<br>
        <span style="color: blue;">Modifié: Sitting in the airport, waiting for the plane to arrive, so we can depart!!  <url></span>


- Utilisation de dictionnaires d'abbréviations, d'expression d'argot et d'emoticons pour interpréter les caractères
- Suppression des caractères spéciaux résiduels (@, #, caractères non ASCII)
- Expansion des contractions et application d'un correcteur d'orthographe (languagetoolPython)
        Original: @DavidArchie Hope you, your team, Cookie &amp; his crew have a safe trip home! You guys are all amazing! Hope you'll get some R&amp;R time now.<br>
        <span style="color: orange;">Modifié: <mention> Hope you, your team, Cookie & his crew have a safe trip home! You guys are all amazing! Hope you will get some Randy time now.</span>
 
L'application de l'ensemble de ces fonctions sur les tweets sélectionnés est rapide grâce à l'utilisation de bibliothèques et aux expressions régulières (3'40").<br>         

### Réduction du vocabulaire

La tokenisation des textes de tweets après les premiers traitements conduit à un vocabulaire de plus de 12 000 termes qu'il faut chercher à réduire pour réduire la taille des matrices.<br>
La suppression de certains signes de ponctuation non informatifs (, . ; :) et une lemmatisation (regroupement des termes ayant la même racine) permet de réduire le vocabulaire de 25%.

## Feature engineering

J'ai utilisé un encodeur étudié spécifiquement pour l'analyse de sentiments. SentimentIntensityAnalyser (SIA de NLTK) attribue un score de sentiment à une une phrase, en combinant simplement les scores de chaque mot de la phrase. Voici un example avec un tweet brut, après nettoyage, après tokenisation/lemmatisation:<br>

![alt text](image-1.png)<br>
![<alt text>](image-2.png)<br>
![alt text](image-4.png)<br>

## Métrique adaptée à la problématique métier

Le client souhaite détecter les "bad buzz" donc les sentiments négatifs en priorité. J'ai donc défini le score de sentiment négatif (initialement 0) comme la classe positive(1) et le score de sentiment positif (initialement 4) comme la classe négative (0).<br>
La métrique principale sera bien sûr l'exactitude globale (accuracy) et pour des performances équivalentes il faudra examiner le rappel (recall) qui est le taux de prédiction positives correctes et donc minimise les faux négatifs.

## Baseline

En comparant la colonne de score SIA aux étiquettes réelles on obtient une accuracy de 0,66. Par contre la matrice de confusion montre que la classe 1 (sentiment négatif) est moins bien prédite que la classe 0.<br>
![alt text](image-5.png)  ![alt text](image-6.png)

# Étape 2 : Modélisation

## Approche classique

### API sur étagère
En première approche j'ai testé le service [Azure AI Language](https://azure.microsoft.com/en-us/products/ai-services/ai-language?msockid=366d561faaeb6ac416084323ab526ba8). La performance est tout juste supérieure à la baseline, avec également une disparité entre les classes. Bien que la documentation du service ne fournisse pas de détail, les modèles utilisés sont basés sur Bert avec des prétraitements. Cela nous prouve que le problème n'est pas trivial !

### Optimisation automatique 

#### AutoML (sans recours au deep learning)

Azure fournit également un service d'optimisation automatique à partir des données textes vers une classification. Le modèle le plus performant est un ensemble constitué de différentes régressions logistiques et de SVM appliqué sur une modélisation du texte par TfIdF. L'exactituce atteinte est de 0,75. <br>
![alt text](image-10.png)<br>
Il est possible de sauvegarder le modèle et le code python utilisé pour sa mise au point, par contre l'environnement nécessaire est complexe et très dépendant de Azure. Néanmoins cet expérimentation nous donne la voie vers le type d'embedding et d'algorithme les plus adaptés à notre problème.

#### Pycaret (on peut utiliser aussi AutoSKLearn)

Pycaret permet d'explorer rapidement un ensemble complet d'algorithmes de classification à partir de jeux de données et possède une fonctionnalité de log automatique dans MLFlow ainsi que l'ensemble des étapes de mise au point d'un modèle à l'aide de commandes simples.<br>
![alt text](image-13.png)
<i> Exploration des algorithmes de classification depuis un embedding TfIdF du texte prétraité </i><br>

Le modèle de stacking combinant Extra Trees, SVM et Régression logistique a les meilleures performances par contre son entrainement 75 fois plus long que les modèles simples comme la régression logistique ; il risque d'être peu réactif en production.<br>
Au final la régression logistique apparait une fois de plus comme une solution intéressante. Une représentation en projection NCA montre que les erreurs sont situées à la frontière entre les classes et non pas aléatoirement réparties<br>
![alt text](image-14.png)

#### Optimisation de la régression logistique

Soyons imaginatif: la régression logistique est plutôt efficace et nous avons par ailleurs l'information de score de sentiment.<br>
Quelques essais de paramètres de la régression logistique et le pipeline est prêt. Même en utilisant la colonne de texte sans pré-traitement les performances sont presque aussi bonnes sur l'échantillon de test que AutoML et un recall de 0,76 sur la classe 1, plutôt bien prédite.<br>
![alt text](image-15.png)   ![alt text](image-16.png)

## Modèle avancé

## Modèle avancé sur mesure

# Étape 3 : Déploiement

## Plan de déploiement

## Interface de tests locale

## Pipeline de déploiement continu

## API déployée

# Étape 4 : Suivi et amélioration

## Performance et incidents

## Détection de prévisions incorrectes

## Mécanisme d'amélioration continue

# Conclusion: Takeouts du projet

