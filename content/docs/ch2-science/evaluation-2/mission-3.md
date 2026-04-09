---
title: "Mission 3. Le Scalpel du Reranker"
weight: 3
---

# Évaluation 2 : L'Architecte du Savoir
## Mission 3 : Le Scalpel du Reranker – Atteindre la précision ultime

Bonjour à toutes et à tous ! Nous approchons du sommet de notre édifice. Grâce à la Mission 2, vous disposez d'un moteur de recherche hybride capable de ramener une liste de candidats pertinents. Mais dans le monde de l'IA de haute précision, être "pertinent" ne suffit pas : il faut être **exact**. 

> [!IMPORTANT]
**Je dois insister :** vos embeddings (Bi-Encoders) sont des sprinteurs, ils parcourent des millions de documents en un clin d'œil, mais ils manquent parfois de finesse. 

Aujourd'hui, nous allons introduire le "Scalpel" : le **Cross-Encoder**. C'est un modèle qui ne se contente pas de comparer des positions GPS, mais qui "lit" réellement l'interaction entre votre question et chaque document. Préparez-vous à voir votre classement bouleversé par la puissance de l'attention totale !

---

## Objectif de la Mission
L'étudiant doit implémenter un pipeline de recherche à deux étages (*Two-stage retrieval*). 

L'enjeu est d'utiliser un modèle **Cross-Encoder** pour ré-évaluer les 10 meilleurs résultats issus de la fusion RRF (Mission 2). L'étudiant devra analyser techniquement comment et pourquoi le modèle de reranking modifie l'ordre final pour corriger les erreurs de "proximité aveugle" des moteurs denses.

---

## Le Concept technique : Bi-Encoder vs Cross-Encoder

Pour réussir cette mission, vous devez comprendre pourquoi nous ajoutons cette couche de complexité. Nous voyons deux schémas fondamentaux pour illustrer cette différence de "cerveau".

### 1. L'architecture à deux étages
La **Figure 9-6 : Pipeline de reranking**  décrit la logistique de votre système.

{{< bookfig src="185.png" week="09" >}}

*   **Le Premier Étage (Search)** : C'est ce que vous avez fait en Mission 2. On fouille dans une base immense (millions de documents) avec des méthodes rapides (FAISS/BM25). On ne cherche pas la perfection, on cherche à ne pas rater la réponse. C'est le "Shortlisting".
*   **Le Deuxième Étage (Rerank)** : On prend une petite poignée de survivants (le Top 10) et on les passe au crible d'un modèle beaucoup plus lourd. 


**Mon intuition** : C'est comme un processus de recrutement. Le premier étage est le tri automatique des CV par mots-clés. Le deuxième étage est l'entretien individuel approfondi avec les 5 meilleurs candidats. L'entretien est long, mais c'est le seul moyen de vérifier l'adéquation réelle.

### 2. L'Attention Totale
La **Figure 8-15 : Cross-encoder reranker** explique la supériorité sémantique du reranker.

{{< bookfig src="186.png" week="09" >}}


*   Dans votre moteur FAISS (Bi-Encoder), la question et le document sont transformés en vecteurs séparément. Ils ne se "rencontrent" qu'à la fin pour un calcul d'angle.
*   Dans le **Cross-Encoder**, comme le montre la figure, la question et le document entrent **ensemble** dans le Transformer.
*   **Le secret technique** : Grâce à la *Self-Attention* (Semaine 3), chaque mot de la question peut interagir avec chaque mot du document. Le modèle peut détecter qu'un petit mot comme "sans" ou "sauf" dans la question rend un document totalement hors-sujet, alors qu'un Bi-Encoder aurait été trompé par la ressemblance globale des autres mots.

---

## 1. Snippets de configuration (Setup technique)
*Nous allons utiliser un modèle de reranking optimisé : le `cross-encoder/ms-marco-MiniLM-L-6-v2`. Il est léger mais extrêmement performant pour classer la pertinence.*

```python
# Installation du moteur de reranking (si non fait précédemment)
# !pip install -q sentence-transformers

from sentence_transformers import CrossEncoder

# Initialisation du Scalpel (Le Reranker)
rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)

# Rappel : ce modèle prend en entrée des PAIRES de textes : [Question, Document]
```

---

## 2. Vos Tâches de Mission

### Tâche 1 : Préparation des paires d'interaction
1.  Récupérez la liste des 10 meilleurs documents (chunks) issus de votre fonction `hybrid_search` de la Mission 2.
2.  Pour chacun de ces documents, créez une liste Python de paires : `[[query, doc_1], [query, doc_2], ..., [query, doc_10]]`.

### Tâche 2 : Le calcul de la pertinence profonde
1.  Passez cette liste de paires dans le modèle `rerank_model.predict()`.

> [!WARNING]
> 2.  **Attention technique** : Les scores des Cross-Encoders ne sont pas des probabilités entre 0 et 1, mais des "Logits" (scores bruts). Ils peuvent être négatifs. Ce qui compte, c'est la valeur relative : plus le score est haut, plus le document est jugé pertinent par rapport à la question.


### Tâche 3 : Le reclassement final
1.  Triez vos documents en fonction du nouveau score fourni par le Cross-Encoder.
2.  Comparez l'ordre obtenu avec celui de la Mission 2. 
**Mon audit** : Identifiez si le document qui était n°1 en Mission 2 est resté n°1 ou s'il a été détrôné.

### Tâche 4 : Analyse de la "Correction Sémantique"
Testez le système complet avec une requête piège, par exemple : *"KV Cache without using GQA"*.
*   Le moteur hybride risque de ramener des documents sur le "GQA" car les mots-clés matchent.
*   Le reranker doit normalement identifier que vous cherchez des solutions **sans** GQA.
*   Rédigez un court paragraphe technique expliquant comment l'attention bidirectionnelle du Cross-Encoder a permis (ou non) de gérer cette négation.

---


## Mes avertissements

> [!WARNING]
>*   **Erreur fréquente :** Essayer de passer 1000 documents au Cross-Encoder. Sur une T4, cela prendrait plusieurs secondes, ce qui est inacceptable pour un utilisateur. **Le reranking est une phase de finition**, elle ne doit traiter que le "haut du panier".
>*   **Attention :** N'oubliez pas de désactiver le calcul des gradients (`with torch.no_grad():`) si vous manipulez le modèle manuellement, afin d'économiser la mémoire de la T4.

> [!TIP]
**Le conseil de l'expert :** Le reranking est le moment où l'IA "réfléchit" vraiment à la relation entre les mots. 
> Si votre reranker ne change jamais l'ordre de votre recherche hybride, c'est soit que votre recherche initiale est parfaite, soit que votre reranker est trop petit pour apporter de la valeur. Testez toujours la sensibilité de votre scalpel !

---

**Une fois que votre pipeline à deux étages est opérationnel, nous passerons à la Mission 4 : l'Évaluation Finale des performances avec les métriques MRR et NDCG.**