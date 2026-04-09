---
title: "Evaluation 2 : L'Architecte du Savoir "
weight: 2
bookCollapseSection: true
---

# Évaluation 2 : L'Architecte du Savoir 
## Construire un moteur de recherche hybride et résilient

Bonjour à toutes et à tous. Nous entamons aujourd'hui une phase charnière de notre cursus technique. Vous avez appris, au cours des dernières semaines, comment les modèles de représentation parviennent à capturer le sens du langage et comment les vecteurs permettent de naviguer dans d'immenses bases de données. Il est désormais temps de passer de la théorie sémantique à l'ingénierie des systèmes de recherche.

> [!IMPORTANT]
**Je dois insister sur une distinction fondamentale** : posséder un modèle puissant est une chose, mais construire un moteur de recherche capable de ramener la vérité documentaire à un utilisateur en est une autre. 

Dans cette épreuve, vous allez endosser le rôle d'un architecte du savoir. Vous allez apprendre que le sens pur (le vecteur) a parfois besoin de la rigueur du mot exact (le lexical) pour être véritablement efficace. Cette évaluation va mettre à l'épreuve votre capacité à assembler des briques technologiques complexes pour créer un système robuste et auditable.

## Structure de l'épreuve

Cette évaluation représente 15 % de votre note finale et se présente sous la forme d'un défi technique intégré dans un notebook Google Colab unique. Vous ne devez pas traiter ces missions comme des exercices isolés, mais comme les étapes successives de la construction d'un produit industriel.

Le projet se décompose en quatre missions interdépendantes :

### Mission 1 : La Forge du Savoir (Ingestion et Chunking)
> Vous devrez mettre en œuvre une stratégie de découpage récursive intelligente. L'enjeu est de préserver l'intégrité sémantique de documents techniques tout en injectant des métadonnées cruciales qui permettront le filtrage futur.

### Mission 2 : Le Cerveau Hybride (Indexation et Fusion RRF)
> Vous construirez un moteur à double entrée mariant FAISS (recherche dense) et BM25 (recherche sparse). Le point critique sera l'implémentation mathématique de la Reciprocal Rank Fusion (RRF) pour réconcilier ces deux signaux.

### Mission 3 : Le Scalpel du Reranker (Précision ultime)
> Vous ajouterez une couche de vérification par attention bidirectionnelle en utilisant un modèle Cross-Encoder. Vous devrez prouver que cette étape parvient à corriger les erreurs de proximité aveugle de votre premier étage de recherche.

### Mission 4 : Le Verdict de la Mesure (Analyse)
> Vous clôturerez votre projet par un audit quantitatif. Vous calculerez le Mean Reciprocal Rank (MRR) de votre système pour justifier vos choix d'architecture par des chiffres indiscutables.

## Détails du barème et de la correction

La notation de ce projet sera basée sur la précision de votre implémentation et la clarté de votre raisonnement :

*   **Logique de Fusion et Algorithmique (30 %)** : La justesse de votre formule RRF et la gestion correcte de l'indexation FAISS (notamment la normalisation L2) seront examinées de près.

*   **Qualité du Traitement de Données (20 %)** : Je noterai la pertinence de votre stratégie de chunking et votre capacité à faire survivre les métadonnées tout au long du pipeline.

*   **Profondeur de l'Analyse Technique (35 %)** : C'est le cœur de l'expertise. Vous devez être capables d'expliquer pourquoi le système a échoué sur certaines requêtes et comment le reranker a redressé la barre. Une interprétation correcte du MRR est indispensable.

*   **Optimisation et Sobriété (15 %)** : Votre aptitude à utiliser efficacement le GPU T4 et à gérer la latence système sera valorisée. Un système précis mais trop lent pour la production sera pénalisé.

## Mes conseils et recommandations

*   Ne négligez pas le premier kilomètre : Si votre découpage de texte est trop brutal en Mission 1, aucune mathématique complexe en Mission 3 ne pourra retrouver l'information perdue.

*   Surveillez vos dimensions : Assurez-vous que la dimension de votre index FAISS correspond exactement à la sortie de votre modèle d'embedding (768 pour MPNet, 384 pour MiniLM).

*   Pensez à l'utilisateur final : Un score de similarité de 0.95 n'est pas une réponse. L'utilisateur veut le document exact. Votre objectif est de mettre la vérité en première position.

*   Révisez votre algèbre linéaire : La compréhension du produit scalaire et de la normalisation vous évitera des résultats de recherche incohérents.


Le succès de cette évaluation réside dans votre capacité à faire dialoguer des outils différents. Soyez méticuleux, soyez logiques et n'oubliez jamais que l'IA est au service de la précision documentaire. Je vous souhaite un excellent travail d'architecture.

