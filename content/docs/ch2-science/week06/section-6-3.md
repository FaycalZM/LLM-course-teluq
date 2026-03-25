---
title: "6.3 Architecture de recherche "
weight: 4
---

## Construire la bibliothèque du futur : Au-delà du vecteur unique

Bonjour à toutes et à tous ! Nous arrivons à un moment charnière. Vous savez maintenant transformer du texte en vecteurs (section 6.1) et mesurer leur distance (section 6.2). Mais imaginez maintenant que vous deviez chercher la réponse à une question dans une base de données contenant **10 millions de documents**. Si vous comparez votre question à chaque document un par un, l'utilisateur aura pris sa retraite avant d'avoir sa réponse ! 

> [!IMPORTANT]
🔑 **Je dois insister :** une bonne recherche sémantique ne repose pas seulement sur l'intelligence du modèle, mais sur la robustesse de l'architecture qui l'entoure. Aujourd'hui, nous allons apprendre à découper, indexer et fouiller massivement.


## L'art délicat du découpage : Le Chunking
Un Transformer a une limite physique : sa fenêtre de contexte. On ne peut pas "donner" un livre entier à BERT pour qu'il en fasse un seul vecteur sans perdre une quantité colossale d'informations. C'est là qu'intervient le **chunking** (le découpage en tronçons).

Regardons ensemble la **Figure 6-8 : Un vecteur par document vs. Plusieurs vecteurs par document**.

{{< bookfig src="178.png" week="06" >}}
 
*   **À gauche (One vector per doc)** : On essaie de résumer tout un article en un point. C'est risqué. Si l'article parle de 10 sujets différents, le vecteur final sera une moyenne floue. 
*   **À droite (Multiple vectors)** : On découpe l'article en petits morceaux cohérents. Chaque morceau a son propre vecteur. C'est la stratégie gagnante pour la précision.

Les stratégies de découpage illustrées dans les **Figures 6-9 à 6-11** sont vos outils de précision :
1.  **Découpage par caractères ou tokens (Figure 6-9)** : On coupe tous les 15 caractères ou 5 tokens. C'est simple mais brutal : on risque de couper une phrase ou une pensée en plein milieu.

{{< bookfig src="179.png" week="06" >}}

2.  **Découpage structurel (Figure 6-10)** : On respecte les paragraphes ou les phrases. On utilise des outils comme `NLTK` ou `Spacy` pour repérer les limites naturelles du langage.

{{< bookfig src="180.png" week="06" >}}

3.  **La fenêtre glissante avec chevauchement (Overlap) (Figure 6-11)** : C'est la technique préférée des experts.
> [!TIP]
🔑 **Notez bien cette intuition :** on crée des morceaux qui se chevauchent (par exemple, les 50 derniers tokens du bloc 1 se retrouvent au début du bloc 2). Pourquoi ? Pour s'assurer que si une information cruciale se trouve à la charnière de deux blocs, le contexte ne soit pas perdu.

{{< bookfig src="181.png" week="06" >}}


## Passer à l'échelle avec FAISS : La recherche de plus proches voisins
Une fois vos millions de "chunks" transformés en vecteurs, comment trouver les plus proches ? On utilise l'algorithme des **K-Plus Proches Voisins (K-Nearest Neighbors ou KNN)**, illustré en **Figure 6-12**.

{{< bookfig src="182.png" week="06" >}}


Pour gérer des millions de vecteurs en millisecondes, nous utilisons **FAISS** (*Facebook AI Similarity Search*), une bibliothèque optimisée pour les calculs matriciels massifs. FAISS nous propose deux mondes :

### 1. La Recherche Exacte (Flat Index)
Le modèle compare votre requête à absolument TOUS les vecteurs. 
*   **Précision** : 100% (on trouve mathématiquement les meilleurs). 
*   **Vitesse** : Lente sur de très grosses bases. 
*   **Usage** : Jusqu'à environ 1 million de documents.

### 2. La Recherche Approximative (ANN - Approximate Nearest Neighbors)
C'est ici que l'ingénierie devient "magique". Au lieu de tout fouiller, on utilise des index intelligents comme **IVF** (Inverted File Index).
*   **L'idée** : On regroupe les vecteurs similaires dans des "cellules" (clusters) à l'avance. Quand une question arrive, on identifie la cellule la plus proche et on ne fouille que celle-là. 
*   **Vitesse** : Foudroyante (recherche en microsecondes parmi des milliards).
*   **Compromis** : On accepte une infime chance de rater le meilleur résultat absolu pour gagner une vitesse immense.


## Architecture complète d'un système de recherche
Un moteur de recherche sémantique robuste suit ce pipeline non-négociable :
1.  **Ingestion** : Lecture des PDF, sites web ou bases de données.
2.  **Preprocessing** : Nettoyage et *Chunking* intelligent.
3.  **Embedding** : Passage dans un modèle comme `all-mpnet-base-v2`.
4.  **Indexing** : Stockage des vecteurs dans FAISS ou une base vectorielle (Chroma, Pinecone).
5.  **Querying** : Encodage de la question de l'utilisateur et recherche KNN.


## Laboratoire de code : Implémentation FAISS sur Colab (T4)
Voici comment construire votre premier index vectoriel professionnel. Nous allons simuler une base de connaissances et effectuer une recherche ultra-rapide.

```python
# Installation : !pip install faiss-cpu sentence-transformers
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Préparation du modèle (all-mpnet-base-v2 produit des vecteurs de taille 768)
model = SentenceTransformer("all-mpnet-base-v2")
dimension = 768 

# 2. Simulation d'une base de données de connaissances
documents = [
    "The capital of France is Paris.",
    "The Pyramids of Giza are located in Egypt.",
    "Python is a popular programming language for AI.",
    "Deep learning is a subset of machine learning.",
    "The Eiffel Tower was completed in 1889."
]

# 3. Encodage des documents
doc_embeddings = model.encode(documents)
# FAISS a besoin de float32 pour fonctionner de manière optimale sur GPU
doc_embeddings = np.array(doc_embeddings).astype('float32')

# 4. Création de l'index FAISS (IndexFlatIP utilise le produit scalaire pour la similarité cosinus)
index = faiss.IndexFlatIP(dimension)
index.add(doc_embeddings) # Ajout des documents à la bibliothèque

# 5. Recherche (Inférence)
query = "Tell me about famous monuments in Europe"
query_embedding = model.encode([query]).astype('float32')

# On cherche les 2 documents les plus proches (k=2)
distances, indices = index.search(query_embedding, k=2)

print(f"Requête : {query}")
print("--- Résultats les plus proches ---")
for i, idx in enumerate(indices[0]):
    print(f"Top {i+1} : {documents[idx]} (Score : {distances[0][i]:.4f})")

```

> [!NOTE]
ℹ️ **Note** : Notez bien l'usage de `IndexFlatIP`. IP signifie *Inner Product*. Comme nos embeddings sont normalisés par `sentence-transformers`, le produit scalaire devient identique à la similarité cosinus. C'est l'astuce de performance préférée des ingénieurs.


## Optimisations matérielles et GPU
Sur votre instance Colab T4, FAISS peut être déplacé sur le GPU pour une accélération encore plus massive. 

> [!IMPORTANT]
🔑 **Je dois insister :** si vous dépassez les 10 millions de vecteurs, le calcul sur CPU deviendra un goulot d'étranglement. L'usage de la mémoire VRAM pour stocker l'index est une compétence clé à développer.


## Éthique et Confidentialité : La persistance des vecteurs

> [!WARNING]
⚠️ Mes chers étudiants, soyez conscients de ce que vous indexez.

> Une fois qu'un document est découpé en morceaux (chunks) et stocké sous forme de vecteurs dans une base de données, il devient très difficile de le "désindexer" totalement, surtout si vous utilisez des techniques de compression. 
> 1.  **Données personnelles (PII)** : Ne stockez jamais de noms, numéros de téléphone ou secrets dans vos index vectoriels sans anonymisation préalable. Un vecteur peut parfois être "renversé" pour retrouver une partie du texte d'origine.
> 2.  **Droit à l'oubli** : Si un utilisateur demande la suppression de ses données, vous devez être capable de retrouver tous les chunks associés à son identité dans votre index FAISS. C'est un défi d'ingénierie légale majeur sous le RGPD.


🔑 **Mon message** : L'architecture de recherche est le système nerveux de votre application. Si vos morceaux sont trop petits, le modèle sera confus. S'ils sont trop gros, il sera imprécis. Trouvez le juste milieu, et n'oubliez jamais que derrière chaque nombre, il y a une donnée humaine qui mérite votre protection.

---
Vous savez maintenant construire le moteur de recherche. Vous savez découper le savoir et l'indexer pour une vitesse fulgurante. Dans la dernière section de cette semaine, nous verrons comment perfectionner ce système pour qu'il ne se contente pas de trouver des documents, mais qu'il apprenne de ses erreurs : **le réglage fin pour le retrieval**.