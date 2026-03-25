---
title: "Laboratoire "
weight: 6
---

Bonjour à toutes et à tous ! Nous y sommes : le moment de transformer votre ordinateur en un bibliothécaire omniscient. Dans ce laboratoire, nous allons construire un moteur de recherche capable de comprendre les intentions cachées derrière les mots. 

> [!TIP]
🔑 **Je dois insister :** ne vous contentez pas de faire tourner le code. Changez les requêtes, observez les scores de similarité, et essayez de "piéger" le modèle. C'est en comprenant ses échecs que vous deviendrez de véritables experts. Prêt·e·s à plonger dans l'espace vectoriel ?


## 🔹 EXERCICE 1 : Moteur de recherche sémantique de base (Niveau 1)

**Objectif** : Implémenter une recherche sémantique simple en comparant manuellement une requête à une liste de documents.

```python
from sentence_transformers import SentenceTransformer, util
import torch

# 1. Chargement du modèle (QUESTION CODE)
model = SentenceTransformer('all-MiniLM-L6-v2') # Modèle très léger pour Colab

documents = [
    "Machine learning is a method of data analysis that automates analytical model building.",
    "The recipe calls for two cups of flour and one cup of sugar.",
    "Deep learning is a subset of machine learning based on artificial neural networks.",
    "Soccer is a sport played between two teams of eleven players."
]

```

<details>
<summary>Voir la réponse</summary>

```python
# --- RÉPONSE (ANSWER CODE) ---
# 2. Encodage des documents et de la requête
doc_embeddings = model.encode(documents, convert_to_tensor=True)
query = "Tell me about neural networks and AI"
query_embedding = model.encode(query, convert_to_tensor=True)

# 3. Calcul de la similarité cosinus via l'utilitaire de SBERT
cosine_scores = util.cos_sim(query_embedding, doc_embeddings)[0]

# 4. Affichage des résultats triés
print(f"Requête : {query}")
top_results = torch.topk(cosine_scores, k=2)

for score, idx in zip(top_results[0], top_results[1]):
    print(f"Score: {score:.4f} | Document: {documents[idx]}")
```

**EXPLICATIONS DÉTAILLÉES**
*   ATTENDU : Le document 3 doit avoir le score le plus élevé car "neural networks" est sémantiquement lié à la requête, même si les mots exacts ne sont pas tous présents.
*   JUSTIFICATION : util.cos_sim gère la normalisation des vecteurs pour nous.

</details>

---

## 🔹 EXERCICE 2 : Indexation scale-up avec FAISS (Niveau 2)

**Objectif** : Utiliser la bibliothèque FAISS pour indexer des documents et effectuer une recherche "K-Plus Proches Voisins" (KNN).

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Préparation (QUESTION CODE)
model = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384 # Dimension spécifique à MiniLM
corpus = ["The moon orbits the Earth.", "The sun is a star.", "Apples are fruits.", "Fast cars are exciting."]

```

<details>
<summary>Voir la réponse</summary>

```python
# --- RÉPONSE (ANSWER CODE) ---
# 2. Encodage et conversion en float32 (exigé par FAISS)
corpus_embeddings = model.encode(corpus)
corpus_embeddings = np.array(corpus_embeddings).astype('float32')

# 3. Initialisation de l'index FAISS (Inner Product pour Cosinus)
index = faiss.IndexFlatIP(dimension)
faiss.normalize_L2(corpus_embeddings) # Normalisation pour que IP == Cosine
index.add(corpus_embeddings)

# 4. Recherche de la requête
query_text = "space and astronomy"
query_embedding = model.encode([query_text]).astype('float32')
faiss.normalize_L2(query_embedding)

# k=1 : On veut le voisin le plus proche
distances, indices = index.search(query_embedding, k=1)

print(f"Résultat FAISS pour '{query_text}' : {corpus[indices[0][0]]}")
print(f"Distance (Similarité) : {distances[0][0]:.4f}")

```
**EXPLICATIONS DÉTAILLÉES**
*   ATTENDU : "The moon orbits the Earth" ou "The sun is a star".
*   JUSTIFICATION : FAISS permet de traiter des millions de documents là où une boucle Python s'effondrerait.L'usage de float32 est crucial pour la compatibilité GPU.

</details>
---


## 🔹 EXERCICE 3 : Évaluation système : Calcul du MRR (Niveau 3)

**Objectif** : Implémenter la métrique Mean Reciprocal Rank (MRR) pour juger la qualité d'un moteur de recherche.

```python
import numpy as np

# 1. Données de test (QUESTION CODE)
# Chaque sous-liste contient les IDs des documents renvoyés par le système.
# 'ground_truth' contient l'ID du SEUL document vraiment pertinent.
predictions = [
    [5, 10, 3, 1], # Requête 1 : le bon doc est l'ID 3 (3ème position)
    [2, 4, 8, 9],  # Requête 2 : le bon doc est l'ID 2 (1ère position)
    [12, 15, 7, 2] # Requête 3 : le bon doc est l'ID 1 (Absent)
]
ground_truths = [3, 2, 1]

```
<details>
<summary>Voir la réponse</summary>

```python
# --- RÉPONSE (ANSWER CODE) ---
def calculate_mrr(preds, targets):
    rr_list = []
    for p, t in zip(preds, targets):
        if t in p:
            rank = p.index(t) + 1 # +1 car les listes commencent à 0
            rr_list.append(1.0 / rank)
        else:
            rr_list.append(0.0)
    return np.mean(rr_list)

mrr_score = calculate_mrr(predictions, ground_truths)
print(f"Score MRR du système : {mrr_score:.4f}")
```
**EXPLICATIONS DÉTAILLÉES**
*   ATTENDU : Score d'environ 0.444 ( (1/3 + 1/1 + 0) / 3 ).
*   JUSTIFICATION : Le MRR est impitoyable. Si le bon document n'est pas 1er, le score chute vite. C'est la métrique reine pour les moteurs de recherche où l'utilisateur ne clique que sur le premier lien.

</details>
---

**Mots-clés de la semaine** : Sentence Embeddings, SBERT, Similarité Cosinus, FAISS, Chunking, Overlap, KNN, Retrieval, MRR, Hit Rate.

**En prévision de la semaine suivante** : Nous allons apprendre à découvrir la structure cachée de vos documents. Comment regrouper automatiquement des milliers de textes par thématiques sans intervention humaine ? Bienvenue dans le monde du **Clustering** et de **BERTopic**.

**SOURCES COMPLÈTES** :
*   Hugging Face : *Semantic Search Guide* (https://huggingface.co/blog/semantic-search).
*   Pinecone : *Vector Similarity* (https://www.pinecone.io/learn/vector-similarity/).
