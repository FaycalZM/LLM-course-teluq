---
title: "1.4 Définition et applications des LLM"
weight: 5
---

## Une définition mouvante : Qu'est-ce qu'un "Large" Language Model ?

Nous y sommes ! Après avoir exploré les briques et le moteur, regardons enfin l'édifice dans son ensemble. Mais attention, le terme "Large Language Model" (LLM) est un peu comme un horizon qui recule à mesure que l'on avance.

Comme l'expliquent Jay Alammar et Maarten Grootendorst, la définition de ce qui est "large" a radicalement changé en quelques années seulement. En 2018, un modèle comme BERT (110 à 340 millions de paramètres) était considéré comme géant. Aujourd'hui, avec des modèles comme GPT-4 qui dépasseraient les mille milliards de paramètres, nos anciens géants ressemblent à des nains. 🔑 **Je dois insister :** Le mot "Large" ne fait pas seulement référence au nombre de paramètres (les "boutons" que le modèle ajuste pendant l'apprentissage), mais aussi à l'immensité des données ingérées : presque tout le texte produit par l'humanité et numérisé sur le web.

Pour ce cours, nous adopterons la définition du livre : un LLM est un modèle de langage capable de comprendre et de générer du texte, entraîné sur des corpus massifs, et qui possède généralement une capacité de généralisation dépassant ses tâches d'entraînement initiales.

## L'épopée GPT : De l'ombre à la lumière

L'histoire des LLM modernes est indissociable de l'évolution de la famille GPT (*Generative Pre-trained Transformer*). Regardez la progression illustrée par les **Figures 1-20 à 1-26** :
1.  **GPT-1 (2018)** : 117 millions de paramètres. C'était la preuve de concept : un Transformer décodeur peut apprendre à lire tout seul.
2.  **GPT-2 (2019)** : 1,5 milliard de paramètres. La rupture ! Le modèle commençait à écrire des articles si crédibles qu'OpenAI a d'abord hésité à le publier par peur des dérives.
3.  **GPT-3 (2020)** : 175 milliards de paramètres. Le moment "Eureka". Sans entraînement spécifique, le modèle pouvait traduire, coder et raisonner simplement grâce au *prompting*.
4.  **2023 : L'explosion** : Comme le montre la **Figure 1-27**, l'année 2023 a marqué une accélération sans précédent avec l'arrivée de Llama (Meta), Falcon (TII), Mistral et bien d'autres, rendant ces puissances de calcul accessibles sur vos propres machines.

{{< bookfig src="25.png" week="01" >}}

{{< bookfig src="26.png" week="01" >}}

{{< bookfig src="27.png" week="01" >}}

{{< bookfig src="28.png" week="01" >}}

{{< bookfig src="29.png" week="01" >}}

{{< bookfig src="30.png" week="01" >}}

{{< bookfig src="31.png" week="01" >}}

{{< bookfig src="32.png" week="01" >}}

## Le paradigme de l'apprentissage : Le secret en deux étapes

C'est ici que vous devez être très attentifs, car c'est la base de votre futur travail d'ingénieur en IA. Un LLM ne naît pas "intelligent", il passe par deux phases distinctes.

1.  **Le Pré-entraînement (Pretraining)** : Imaginez un étudiant qui lirait toutes les bibliothèques du monde pendant 20 ans, mais sans professeur. Il connaît tout, il sait prédire le mot suivant avec une précision diabolique, mais il n'est pas "poli" et ne sait pas forcément répondre à une question. Il complète simplement la séquence. On appelle cela un **Foundation Model** ou **Base Model**.
2.  **Le Réglage Fin (Fine-tuning / Instruction Tuning)** : C'est l'étape où l'on donne un "professeur" au modèle. On lui montre des exemples de dialogues, de questions-réponses et de comportements souhaités. C'est ce qui transforme un prédicteur de texte brut en un assistant comme ChatGPT ou Claude.

{{< bookfig src="34.png" week="01" >}}

{{% hint warning %}}
**Attention : erreur fréquente ici !** Beaucoup d'utilisateurs pensent que le modèle "apprend" de nouvelles informations pendant qu'ils lui parlent. En réalité, le modèle est "gelé". Il utilise ses connaissances acquises lors du pré-entraînement pour traiter votre demande actuelle.
{{% /hint %}}

## Applications pratiques : Un couteau suisse universel

Le champ d'application des LLM est si vaste qu'il redéfinit des industries entières. Voici un aperçu des tâches qu'un LLM peut accomplir sans être spécifiquement programmé pour elles :

**Tableau 1-2 : Applications typiques des LLM**

| Domaine | Exemple de tâche | Valeur ajoutée |
| :--- | :--- | :--- |
| **Rédaction** | Copywriting, emails, articles | Gain de productivité massif |
| **Analyse** | Résumé de documents longs, extraction d'entités | Gain de temps d'examen |
| **Code** | Génération de fonctions Python, débogage | Aide aux développeurs (Copilot) |
| **Sémantique** | Recherche d'information par le sens (pas par mot-clé) | Moteurs de recherche intelligents |
| **Créativité** | Aide à l'idéation, scénarisation | Partenaire de brainstorming |

## Éthique et limites : Garder les yeux ouverts

{{% hint danger %}}
Je ne serais pas une bonne enseignante si je ne vous mettais pas en garde. Ces modèles sont des prouesses technologiques, mais ils ont des failles profondes que vous devez gérer.

1.  **Hallucinations** : Comme le modèle ne fait que prédire le mot "statistiquement le plus probable", il peut inventer des faits, des dates ou des citations juridiques avec un aplomb total. 🔑 **Je dois insister :** Ne faites jamais une confiance aveugle à la sortie d'un LLM sans vérification.
2.  **Biais et représentations** : Le modèle est le miroir de ses données. S'il a lu des textes biaisés, il produira des réponses biaisées. La neutralité de l'IA est un mythe ; la responsabilité de l'humain est une réalité.
3.  **Transparence et opacité** : Nous sommes face à des "boîtes noires". Expliquer pourquoi un modèle a pris telle décision est l'un des plus grands défis de la recherche actuelle.
{{% /hint %}}

{{% hint info %}}
Vous n'apprenez pas seulement à utiliser des outils, vous apprenez à dompter une puissance statistique immense. L'éthique n'est pas une option, c'est le garde-fou qui sépare une innovation utile d'un désastre sociétal.
{{% /hint %}}

Nous avons terminé notre tour d'horizon théorique ! Vous avez maintenant une vision claire de la forêt. Dès la semaine prochaine, nous allons nous approcher des arbres et examiner les feuilles : les tokens et les embeddings. Mais d'abord, place à la pratique en laboratoire !

## **SOURCES COMPLÈTES**

- Jay Alammar : The Illustrated Transformer ([https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/))
- Maarten Grootendorst : LLM Roadmap 2023 ([https://maartengr.github.io/2023/12/19/llm-roadmap.html](https://maartengr.github.io/2023/12/19/llm-roadmap.html))
- Hugging Face : NLP Course - Introduction ([https://huggingface.co/learn/nlp-course/chapter1/1](https://huggingface.co/learn/nlp-course/chapter1/1))
- GitHub Officiel : [https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/tree/main/chapter01](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/tree/main/chapter01)
