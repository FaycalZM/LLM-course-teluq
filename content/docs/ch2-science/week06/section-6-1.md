---
title: "6.1 Embeddings de phrases "
weight: 2
---

## Le défi du passage à l'échelle sémantique
> [!NOTE]
Mes chers étudiants, rappelez-vous de notre [**Semaine 2**]({{< relref "section-2-1.md" >}}). 

Nous avions appris que chaque mot peut être un vecteur. Mais dans la vie réelle, nous ne communiquons pas par mots isolés. Nous utilisons des phrases, des paragraphes, des intentions. 

Si vous essayez de comparer deux documents en faisant simplement la moyenne des vecteurs de chaque mot (Word2Vec), vous obtiendrez un résultat médiocre. Pourquoi ? Parce que la syntaxe et l'ordre des mots comptent ! "L'homme a mordu le chien" et "Le chien a mordu l'homme" ont les mêmes mots, mais des sens radicalement opposés. Pour résoudre cela, nous avons besoin d'**embeddings de phrases** (*Sentence Embeddings*). 

Comme l'illustre la **Figure 6-1 : Processus de création d'embeddings textuels**, le but est d'utiliser un modèle de langage comme un extracteur de caractéristiques globales. Le modèle reçoit une phrase entière et, après être passé par ses couches de Transformers, il condense toute l'information dans un vecteur unique de taille fixe (souvent *384*, *768* ou *1024* dimensions).

{{< bookfig src="46.png" week="06" >}}


## La naissance de Sentence-BERT (SBERT)
Pendant longtemps, utiliser BERT pour comparer des phrases était incroyablement lent. Comme nous l'avons vu en Semaine 4, BERT est conçu pour la classification ou le remplissage de masques. Pour comparer deux phrases, il fallait les passer ensemble dans le modèle (*Cross-Encoder*), ce qui était impossible à grande échelle.

La révolution est venue de **SBERT** (Reimers & Gurevych, 2019). 
> [!IMPORTANT]
🔑 **Je dois insister sur cette innovation :** SBERT utilise une architecture dite "Siamese" (Siamoise).

> On utilise deux réseaux BERT identiques qui partagent les mêmes poids. Chaque phrase passe par un réseau, et on optimise le modèle pour que les phrases ayant un sens proche finissent avec des vecteurs proches. 

C'est ce que montre la **Figure 6-2 : Recherche sémantique dense**. Dans cette illustration, vous pouvez voir que dans l'espace mathématique du modèle, la phrase "Le ciel est bleu" (text 1) se retrouve physiquement proche de "Il fait beau aujourd'hui" (text 2), alors qu'elle est très éloignée de "Le moteur de ma voiture est cassé" (text 3). 

{{< bookfig src="175.png" week="06" >}}

> [!NOTE]
🔑 **Notez bien cette intuition :** La recherche sémantique transforme le langage en une carte géographique où la proximité physique égale la proximité de sens.


## L'art du Pooling : Comment résumer une séquence ?
Un Transformer produit un vecteur pour *chaque* token de la phrase. Si votre phrase fait 10 tokens, vous avez 10 vecteurs. Comment n'en obtenir qu'un seul pour représenter la phrase entière ? C'est ce qu'on appelle le **Pooling**.

Il existe trois stratégies principales :
1.  **CLS Pooling** : On utilise uniquement le vecteur du premier token spécial `[CLS]`. C'est la méthode historique de BERT.
2.  **Mean Pooling (Moyenne)** : On calcule la moyenne mathématique de tous les vecteurs de la phrase. 
>> [!TIP]
🔑 **C'est la méthode recommandée par SBERT :** Elle capture mieux l'essence globale de la séquence que le simple token `[CLS]`.
3.  **Max Pooling** : On prend la valeur maximale de chaque dimension à travers tous les tokens.

> [!WARNING]
⚠️ **Attention : erreur fréquente ici !** N'essayez pas de faire du pooling manuellement si vous utilisez la bibliothèque `sentence-transformers`. Elle gère cela automatiquement selon le modèle choisi.


## Choisir son modèle : Modèles de fondation vs Modèles spécialisés

> [!IMPORTANT]
🙅‍♂️ Ne prenez pas le premier modèle venu sur Hugging Face !

Pour la recherche sémantique, la taille ne fait pas tout: 

*   **all-mpnet-base-v2** : C'est le "couteau suisse (swiss army knife)". Avec ses 768 dimensions, il offre le meilleur équilibre entre précision et vitesse pour l'anglais.
*   **paraphrase-multilingual-MiniLM-L12-v2** : Indispensable si vous travaillez en français ou dans plusieurs langues simultanément. Il est très léger et rapide.
*   **gte-small / bge-micro** : Des modèles très récents (2023-2024) qui battent des records sur les benchmarks tout en étant minuscules.

> [!TIP]
🔑 **Le baromètre absolu : Le MTEB Leaderboard**. Comme nous l'avons évoqué, le *Massive Text Embedding Benchmark* est votre boussole. Avant de déployer un système, vérifiez la position de votre modèle dans la colonne "Retrieval" (Recherche).


## Projection spatiale et intuition de recherche

Regardez la **Figure 6-3 : Projection de la requête**. Lorsqu'un utilisateur pose une question, le système transforme cette question en un vecteur (une étoile dans la galaxie). Le moteur de recherche n'a plus qu'à regarder quelles sont les "étoiles" (les documents) les plus proches de la question.

{{< bookfig src="176.png" week="06" >}}


C'est ce qui est détaillé en **Figure 6-4 : Workflow de la base de connaissance** : 
1.  On transforme tous nos documents en vecteurs à l'avance (Indexation).
2.  On les stocke dans une "base de données vectorielle".
3.  À la requête, on compare et on affiche les résultats.

{{< bookfig src="177.png" week="06" >}}


## Laboratoire de code : Créer ses premiers embeddings (Colab T4)
Voici comment implémenter cela très simplement. Nous allons utiliser la bibliothèque `sentence-transformers`, qui est le standard de l'industrie.

```python
# Installation : !pip install sentence-transformers
from sentence_transformers import SentenceTransformer
import torch

# 1. Chargement du modèle de pointe
model = SentenceTransformer("all-mpnet-base-v2", device="cuda")

# 2. Nos phrases à comparer
sentences = [
    "The cat is relaxing on the sofa.",
    "A feline is resting on the couch.", # Très proche sémantiquement
    "The stock market experienced a significant drop today." # Très éloigné
]

# 3. Encodage : Transformation en vecteurs (Embeddings)
embeddings = model.encode(sentences)

print(f"Forme de la matrice d'embeddings : {embeddings.shape}")
# Attendu : (3, 768) -> 3 phrases de 768 dimensions chacune.

# 4. Vérification de la dimension
print(f"Dimension du vecteur pour une phrase : {len(embeddings[0])}")
# Attendu : 768

# 5. Comparaison des embeddings
from sklearn.metrics.pairwise import cosine_similarity

print(f"Similarité entre phrase 1 et 2 : {cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]}")
print(f"Similarité entre phrase 1 et 3 : {cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]}")

```

> [!NOTE]
🔎 Observez bien la sortie du code. La phrase 1 et la phrase 2 n'ont presque aucun mot en commun ("cat" vs "feline", "sofa" vs "couch"). Pourtant, grâce aux embeddings de phrases, leurs vecteurs seront presque identiques. C'est ici que vous tuez la recherche par mot-clé !

## Éthique et Transparence : Les dimensions cachées du sens

> [!WARNING]
⚠️ Mes chers étudiants, un vecteur est une réduction de la réalité. 
Quand vous compressez une pensée humaine complexe dans 768 nombres, vous perdez forcément des nuances. 
1.  **Biais de distance** : Si votre modèle a été entraîné sur des textes majoritairement occidentaux, il pourrait considérer que "mariage" est plus proche de "fête" que de "contrat", ce qui n'est pas universel.
2.  **Sensibilité aux détails** : Les embeddings de phrases sont parfois "trop globaux". Ils peuvent rater une petite négation ("ne... pas") qui change tout le sens de la phrase. 

> [!IMPORTANT]
🔑 **Je dois insister :** La recherche sémantique est un outil de découverte, pas un juge de vérité. Toujours prévoir une étape de vérification humaine ou un modèle de reranking (que nous verrons en [**section 6.3**]({{< relref "section-6-3.md" >}})) pour affiner les résultats.

---
Vous avez maintenant les bases de la cartographie sémantique. Vous savez transformer du texte en coordonnées. Dans la section suivante ➡️, nous allons apprendre à mesurer avec précision la "distance" entre ces points : bienvenue dans le monde de la **similarité cosinus**.