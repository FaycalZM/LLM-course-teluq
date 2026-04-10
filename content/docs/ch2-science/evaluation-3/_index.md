---
title: "Evaluation 3 : Le Gardien de la Vérité "
weight: 5
bookCollapseSection: true
---

# Évaluation 3 : Le Gardien de la Vérité

Bonjour à toutes et à tous. Nous entamons aujourd'hui l'aboutissement de notre deuxième grand chapitre technique. Lors de l'évaluation précédente, vous avez construit un moteur capable de retrouver des documents. Aujourd'hui, nous allons lui donner une voix, mais une voix sous haute surveillance.

> [!IMPORTANT]
**Je dois insister sur un point qui définit l'éthique de notre métier** : une intelligence artificielle qui a du style mais qui invente ses propres faits est un danger pour l'utilisateur et une faute professionnelle pour l'ingénieur. 

Dans cette épreuve, vous allez construire un système de Génération Augmentée par Récupération (RAG) complet. Votre mission est de transformer un modèle génératif en un expert de support technique pour une entreprise de Cloud Computing. Vous allez apprendre à ancrer la parole de la machine dans une base de connaissances stricte et à forcer l'IA à prouver chaque affirmation par une citation vérifiable. Nous passons de la simple discussion au contrat de preuve.


## Structure de l'épreuve

Cette évaluation représente 15 % de votre note finale. Elle se déroule dans un notebook Google Colab unique et se compose de quatre missions qui forment un pipeline de production cohérent. Notez que pour cette épreuve, votre base de connaissances technique est intégralement rédigée en anglais pour refléter les standards de l'industrie internationale.

### Mission 1 : Ingestion Sécurisée (Filtrage par permissions)
> Vous devrez implémenter un mécanisme de filtrage par métadonnées. L'IA doit être physiquement incapable de "voir" des documents confidentiels si l'utilisateur ne possède pas le niveau d'accréditation requis. La sécurité doit être intégrée au niveau du Retriever et non après la recherche.

### Mission 2 : Le Pont du Langage (Query Expansion)
> Vous utiliserez un LLM intermédiaire pour traduire les requêtes floues ou imprécises des utilisateurs en un faisceau de trois requêtes techniques optimisées. L'objectif est d'augmenter le rappel sémantique du système.

### Mission 3 : Le Pilier de la Preuve (Grounded Generation)
> Vous concevrez un prompt système avancé interdisant formellement au modèle d'utiliser ses connaissances internes. Le modèle devra synthétiser une réponse en citant obligatoirement ses sources entre crochets. Vous devrez utiliser un décodage déterministe pour garantir la fiabilité.

### Mission 4 : L'Audit de Confiance (Framework Ragas)
> Vous clôturerez l'épreuve par une évaluation scientifique automatisée. En utilisant l'IA pour juger l'IA, vous calculerez les scores de Faithfulness (Fidélité au contexte) et d'Answer Relevancy pour valider votre déploiement.


## Détails du barème et de la correction

La notation évaluera votre capacité à construire un système "étanche" et auditable :

*   **Rigueur de la Sécurité et du Filtrage (25 %)** : Le respect strict des niveaux d'accès (public/interne/confidentiel) est éliminatoire s'il n'est pas respecté.

*   **Logique de Récupération et Expansion (20 %)** : La pertinence des variations de requêtes générées par le modèle de réécriture sera analysée.

*   **Qualité de l'Ancrage et Citations (30 %)** : Je vérifierai si chaque affirmation de la réponse finale est effectivement présente dans les documents sources cités. Toute hallucination non détectée par votre prompt sera pénalisée.

*   **Audit Scientifique (25 %)** : Votre capacité à interpréter les scores Ragas et à identifier les points de faiblesse du pipeline (ex: une fidélité parfaite mais une utilité faible) sera déterminante pour la note maximale.


## Conseils et recommandations du Professeur Henni

*   Utilisez le froid mathématique : Pour la Mission 3, fixez votre température à 0.0. En RAG industriel, nous ne cherchons pas la poésie mais l'exactitude. La reproductibilité est votre alliée.

*   Délimitez votre contexte : Soyez extrêmement clairs dans votre prompt système sur ce que le modèle a le droit de faire. Utilisez des balises XML ou des titres en majuscules pour séparer les instructions des documents.

*   Surveillez la latence : L'expansion de requête (Mission 2) ajoute un appel au modèle. Assurez-vous que vos questions générées restent concises pour ne pas ralentir inutilement l'expérience utilisateur.

*   Ne négligez pas le refus : Un bon système RAG est un système qui sait dire "Je ne sais pas" lorsque l'information manque. Testez votre modèle avec des questions pièges hors-contexte.


Construire un RAG est un acte de responsabilité. Vous donnez à une machine le pouvoir de parler au nom de votre entreprise. Soyez les gardiens vigilants de cette vérité. Je vous souhaite une excellente session de travail.