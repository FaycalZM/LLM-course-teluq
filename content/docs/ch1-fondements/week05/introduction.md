---
title: "Introduction "
weight: 1
---

## GPT et l'art de la génération textuelle

Bonjour à toutes et à tous ! Je suis ravie de vous retrouver pour cette étape charnière de notre parcours. La semaine dernière, nous avons étudié BERT, un modèle qui excelle dans l'écoute et la compréhension. Aujourd'hui, nous basculons de l'autre côté du miroir : 
> [!TIP]
nous allons explorer les modèles qui "parlent" et qui "créent".

Préparez-vous, car l'architecture **Decoder-only** et la famille **GPT** sont les moteurs de la révolution créative que nous vivons actuellement. C'est fascinant, n'est-ce pas ?

**Rappel semaine précédente** : La semaine dernière, nous avons maîtrisé les modèles de représentation (Encoder-only) comme BERT, en apprenant comment le token [CLS] et le pré-entraînement par masquage (MLM) permettent de classer le texte avec une précision chirurgicale.

**Objectifs de la semaine** :
À la fin de cette semaine, vous saurez :
>*   Expliquer l'architecture interne des modèles *Decoder-only*.
>*   Décrire le processus de prédiction du prochain token et la nature autorégressive.
>*   Tracer l'évolution de la lignée GPT, de GPT-1 à GPT-4.
>*   Distinguer un modèle de fondation (*Foundation Model*) d'un modèle réglé pour les instructions (*Instruction-tuned*).
>*   Paramétrer une génération de texte pour contrôler la créativité et le déterminisme.
