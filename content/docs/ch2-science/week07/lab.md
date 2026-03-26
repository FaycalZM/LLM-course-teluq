---
title: "Laboratoire"
weight: 6
---

Bonjour à toutes et à tous ! Nous y sommes : le moment de transformer une montagne de documents illisibles en une carte thématique structurée. Dans ce laboratoire, nous allons mettre en pratique notre "vision panoramique". 

> [!IMPORTANT]
🔑 **Je dois insister :** ne vous contentez pas de regarder les jolies bulles colorées. Un bon scientifique de la donnée est celui qui va fouiller dans les "outliers" (les points gris) pour comprendre ce que la machine n'a pas réussi à classer. Prêt·e·s à organiser le chaos ? C'est parti !

---

## 🔹 EXERCICE 1 : Pipeline de clustering complet (Niveau 1)

**Objectif** : Implémenter manuellement le pipeline (Embedding -> UMAP -> HDBSCAN) pour découvrir des structures dans un petit corpus.

```python
# --- (QUESTION) ---
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
import numpy as np

docs = [
    "The James Webb telescope sends incredible images of galaxies.",
    "Space exploration requires advanced propulsion systems.",
    "Baking a cake requires flour, sugar, and eggs.",
    "Perfecting a chocolate soufflé is an art in pastry.",
    "NASA is planning a new mission to the moon's south pole."
]

# TÂCHE : Transformez les textes en embeddings, réduisez les dimensions et créez les clusters.

```


<details>
<summary>Voir la réponse</summary>

```python
# --- RÉPONSE (CORRIGÉ) ---
# 1. Embeddings (On utilise un modèle léger pour la démo)
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(docs)

# 2. Réduction de dimension (UMAP)
# On réduit à 2 dimensions pour pouvoir visualiser (exercice simplifié)
umap_model = UMAP(n_neighbors=2, n_components=2, min_dist=0.0, metric='cosine', random_state=42)
reduced_embeddings = umap_model.fit_transform(embeddings)

# 3. Clustering (HDBSCAN)
# min_cluster_size=2 : On veut des thèmes d'au moins 2 documents
cluster_model = HDBSCAN(min_cluster_size=2, metric='euclidean')
clusters = cluster_model.fit_predict(reduced_embeddings)

print(f"Clusters assignés : {clusters}")
# ATTENDU : Des clusters regroupant docs 0, 1, 4 (Espace) et docs 2, 3 (Cuisine).
```
**Explications détaillées** :
*   **Résultats attendus** : Le modèle doit séparer les phrases sur l'espace de celles sur la cuisine.
*   **Justification** : Même avec 2 dimensions (UMAP), la séparation sémantique est si forte que HDBSCAN identifie les deux densités distinctes. Les "outliers" (si présents, notés -1) indiqueraient une phrase trop différente des autres.

</details>


---

## 🔹 EXERCICE 2 : BERTopic avancé avec Reranking (Niveau 2)

**Objectif** : Utiliser la modularité de BERTopic pour affiner la représentation des sujets avec KeyBERTInspired.

```python
# --- (QUESTION) ---
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired

# Supposons que nous avons une liste 'abstracts' de 1000 articles ArXiv
# abstracts = [...] 

# TÂCHE : Initialisez BERTopic et mettez à jour les étiquettes avec KeyBERTInspired.

```

<details>
<summary>Voir la réponse</summary>

```python
# --- RÉPONSE (CORRIGÉ) ---
# 1. Création du modèle de base
topic_model = BERTopic(embedding_model="all-MiniLM-L6-v2", verbose=True)
topics, probs = topic_model.fit_transform(abstracts)

# 2. Sculpture sémantique avec KeyBERTInspired (Reranking)
representation_model = KeyBERTInspired()

# 3. Mise à jour SANS recalculer les clusters (gain de temps !)
topic_model.update_topics(abstracts, representation_model=representation_model)

print("Nouveaux mots-clés (plus sémantiques) :")
print(topic_model.get_topic(0)[:5])

```
**Explications détaillées** :
*   **Résultats attendus** : Les mots-clés du sujet 0 doivent être plus précis (ex: "neural_networks" au lieu de "model").
*   **Justification** : `update_topics` permet de changer la "lentille" à travers laquelle on regarde les clusters existants. KeyBERTInspired recalcule la proximité entre les mots et le centre du cluster.

</details>

---

## 🔹 EXERCICE 3 : Visualisation et gestion du bruit (Niveau 3)

**Objectif** : Générer une visualisation et analyser le cluster des "outliers".

**Tâche** : 
1. Exécutez `topic_model.visualize_hierarchy()` sur vos résultats de l'exercise 2.
2. Identifiez le nombre de documents classés en `-1` (bruit) via `topic_model.get_topic_info()`.

<details>
<summary>Voir la réponse</summary>

**Réponse typique et analyse** :
*   **Action** : `topic_model.visualize_hierarchy()` affiche un dendrogramme montrant comment les sujets se regroupent.
*   **Interprétation du bruit (-1)** : 
>> [!WARNING]
⚠️ Si plus de 30% de vos documents sont en `-1`, cela signifie que vos paramètres `min_cluster_size` sont trop stricts ou que vos données sont trop disparates. 
*   **Justification** : BERTopic privilégie la "pureté" d'un sujet. Il préfère ne rien dire sur un document plutôt que de l'inclure de force dans un thème qui ne lui correspond pas.

</details>

---

**Mots-clés de la semaine** : Clustering, UMAP, HDBSCAN, BERTopic, c-TF-IDF, Outliers, MMR, KeyBERTInspired, Visualisation hiérarchique, Représentation sémantique.

**En prévision de la semaine suivante** : Nous allons apprendre à maîtriser l'interface entre l'humain et la machine : l'art du **Prompt Engineering**. Comment formuler vos demandes pour obtenir le meilleur des LLM ?

**SOURCES COMPLÈTES** :
*   BERTopic Documentation : https://maartengr.github.io/BERTopic/
*   SBERT Clustering Guide : https://www.sbert.net/examples/applications/clustering/README.html
