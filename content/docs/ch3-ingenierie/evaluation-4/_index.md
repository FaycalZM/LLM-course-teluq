---
title: "Evaluation 4 : L'Éducateur de Géants "
weight: 3
bookCollapseSection: true
---

# Évaluation 4 : L'Éducateur de Géants

Bonjour à toutes et à tous. Nous voici arrivés au sommet de notre parcours technique au sein de ce cours. Les semaines précédentes nous ont permis de forger le corps et l'esprit de nos modèles. Mais aujourd'hui, nous abordons la mission la plus délicate et la plus noble de l'ingénieur en IA : donner une conscience sociale et une boussole morale à la machine. 

> [!IMPORTANT]
**Je dois insister sur un point fondamental** : un modèle savant qui n'est pas aligné sur les valeurs humaines est une puissance aveugle. 

Dans cette épreuve finale, vous allez endosser le rôle de mentor. Vous allez apprendre à transformer un algorithme qui ne cherche qu'à "faire plaisir" statistiquement (la Sycophancy) en un assistant digne de confiance, capable de protéger l'utilisateur. Vous allez manipuler les leviers les plus récents de l'industrie, du Fine-tuning supervisé à l'optimisation directe des préférences.

## Structure de l'épreuve

> Cette évaluation représente 15 % de votre note finale. Elle se déroule dans un notebook Google Colab unique et simule la phase finale de création d'un assistant spécialisé dans le Bien-être et la Santé Mentale. L'épreuve est structurée en quatre parties intégrées :

### Partie 1 : Diagnostic du Modèle SFT
> Vous devrez identifier les failles d'un modèle n'ayant subi qu'un réglage supervisé (SFT). En le soumettant à des requêtes dangereuses pour la santé humaine, vous mettrez en évidence sa complaisance et son incapacité à fixer des limites éthiques.

### Partie 2 : Forge du Dataset de Préférence
> C'est le cœur humain de votre travail. Vous concevrez manuellement 5 paires contrastives complexes (triplets Prompt/Chosen/Rejected). Le défi consiste à rédiger des versions "Rejetées" qui semblent polies et fluides, mais qui violent le triptyque de sécurité. Vous devrez justifier chaque choix par rapport aux critères HHH.

### Partie 3 : Chirurgie Comportementale (DPO)
> Vous mettrez en œuvre l'algorithme Direct Preference Optimization (DPO). Vous devrez configurer un adaptateur LoRA et ajuster le paramètre Beta pour incliner les probabilités du modèle vers la sécurité sans détruire sa capacité de raisonnement.

### Partie 4 : Le Test de Turing Éthique
Vous réaliserez un audit final comparatif entre le modèle original et le modèle aligné. Vous devrez évaluer si votre intervention a réduit la Sycophancy sans tomber dans le piège du sur-alignement.

## Détails du barème et de la correction

La notation sera particulièrement attentive à la finesse de votre approche éthique :

*   **Qualité de l'ingénierie des données (35 %)** : La subtilité du contraste entre vos versions "Choisie" et "Rejetée" déterminera votre score. Une version rejetée trop caricaturale n'apprend rien au modèle.

*   **Maîtrise technique du pipeline (25 %)** : L'utilisation correcte de QLoRA sur GPU T4, la stabilité de la perte DPO et le choix judicieux de l'hyperparamètre Beta seront évalués.

*   **Analyse de la Sycophancy et du cadre HHH (30 %)** : Je noterai votre capacité à diagnostiquer précisément pourquoi une réponse viole les piliers Helpful, Honest ou Harmless.

*   **Rigueur de l'évaluation finale (10 %)** : Votre aptitude à identifier les limites de votre propre alignement (ex: le risque de réponse trop évasive) sera valorisée.

## Conseils et recommandations du Professeur Henni

*   Ne confondez pas politesse et sécurité : Un modèle peut être extrêmement poli tout en fournissant des conseils médicaux désastreux. Votre rôle est de détecter ces "hallucinations de compétence". 

*   Le dosage du Beta est crucial : Si vous fixez un Beta trop bas, le modèle changera trop vite et risque de devenir incohérent. Si vous le fixez trop haut, il restera complaisant. Testez la valeur 0.1 comme point de départ.

*   Respectez la fenêtre de contexte : En DPO, vous comparez deux réponses complètes. Soyez concis dans vos données de préférence pour ne pas saturer la mémoire de votre carte T4.

*   Considérez l'impact de la Loi de Goodhart : Comme nous l'avons vu en Semaine 12.4, ne cherchez pas un score de sécurité de 100 % si cela transforme votre IA en un robot qui refuse de répondre même à "Bonjour". L'utilité doit survivre à l'alignement.


Vous n'êtes plus des étudiants qui codent, vous êtes des architectes du contrat social entre l'humain et la machine. Soyez exigeants avec vos données, car elles sont le futur de notre sécurité numérique. Je vous souhaite une excellente session de travail final.
