---
title: "Mission 2. Le Cerveau Hybride"
weight: 2
---

{{< katex />}}

# Évaluation 2 : L'Architecte du Savoir
## Mission 2 : Le Cerveau Hybride – FAISS et la Fusion de Scores RRF

Bonjour à toutes et à tous ! Félicitations pour avoir forgé vos documents en Mission 1. Mais attention, posséder des documents bien découpés est inutile si votre moteur de recherche est "borgne". 

> [!IMPORTANT]
**Je dois insister :** la recherche vectorielle (Dense) est magique pour saisir les concepts, mais elle est parfois terriblement médiocre pour trouver des termes techniques exacts comme "GQA" ou "NF4". 

À l'inverse, la recherche par mots-clés (Sparse/BM25) est d'une précision chirurgicale sur les noms, mais ignore totalement le sens. Aujourd'hui, nous allons construire un cerveau hybride. Nous allons marier ces deux mondes grâce à une technique mathématique élégante : la **Reciprocal Rank Fusion (RRF)**. Préparez-vous, nous allons apprendre à l'IA à ne plus choisir entre le fond et la forme !

---

## Objectif de la Mission
L'étudiant doit transformer les chunks de la Mission 1 en un système de recherche à deux moteurs. L'enjeu est d'implémenter l'algorithme **RRF** pour fusionner les classements denses (Vecteurs) et parses (BM25), créant ainsi une liste de résultats robuste face aux requêtes ambiguës.

---

## 1. Snippets de configuration (Setup technique)
*Utilisez ces briques pour initialiser vos modèles. Nous utilisons `IndexFlatIP` car nos vecteurs seront normalisés, rendant le produit scalaire identique à la similarité cosinus.*

```python
# Initialisation du modèle d'embedding (Cerveau Sémantique)
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
dimension = 384 # Dimension de MiniLM-L6

# Préparation pour BM25 (Cerveau Lexical)

# Note : Vous devrez extraire le texte brut de vos chunks de la Mission 1 
# pour alimenter BM25.
corpus_text = [chunk.page_content for chunk in chunks]
```

---

## 2. Vos Tâches de Mission

### Tâche 1 : L'Indexation Dense (FAISS)
1.  Transformez tous les `chunks` de la Mission 1 en embeddings.
2.  Normalisez les vecteurs (L2) pour permettre l'usage de la similarité cosinus via le produit scalaire.
3.  Créez un index FAISS `IndexFlatIP` et injectez-y vos vecteurs.


### Tâche 2 : L'Indexation Sparse (BM25)
1.  Tokenisez le texte des chunks (découpage par mots simples).
2.  Initialisez l'objet `BM25Okapi` avec ce corpus. 

> [!NOTE]
**Note** : C'est ce moteur qui sauvera votre recherche si l'utilisateur tape un sigle technique rare que l'embedding a mal "digéré".


### Tâche 3 : La Fusion de Scores RRF
C'est le cœur de l'évaluation. Vous devez écrire une fonction `hybrid_search(query, k=5)` qui :
1.  Récupère les **Top 10** résultats du moteur FAISS.
2.  Récupère les **Top 10** résultats du moteur BM25.
3.  Applique la formule **RRF** pour recalculer un score global pour chaque document présent dans l'une ou l'autre liste.
    *   **Formule RRF** : $Score(d) = \sum_{m \in Moteurs} \frac{1}{k + Rank_{m}(d)}$
    *   Utilisez la constante standard $k = 60$. [SOURCE: CONCEPT À SOURCER – Documentation technique Elasticsearch/LangChain sur RRF]


### Tâche 4 : Audit de la Fusion
Testez votre fonction avec la requête : *"How to reduce LLM latency?"*.
Affichez le score final RRF et la source (metadata) du document n°1.

---

## Avertissements du Professeur Henni

> [!WARNING]
>*   **Erreur fréquente :** Essayer d'additionner directement les scores FAISS (ex: 0.85) et les scores BM25 (ex: 15.4). C'est comparer des pommes et des oranges ! **Le RRF ne regarde que le RANG (la position)**, ce qui neutralise la différence d'échelle.
>*   **Attention :** Pour FAISS, l'ID renvoyé par `index.search` est l'index de votre liste de chunks. Assurez-vous de bien faire le lien pour récupérer les métadonnées de la Mission 1.

> [!TIP]
**Le conseil de l'expert :** Le paramètre $k=60$ dans le RRF sert à éviter que les premiers résultats d'un moteur n'écrasent totalement les bons résultats d'un autre moteur qui seraient un peu plus bas. C'est le garant de l'équilibre démocratique de votre recherche !

---

**Une fois que votre cerveau hybride est fonctionnel, nous passerons à la Mission 3 : le Reranking avec un Cross-Encoder pour atteindre la précision ultime.**
