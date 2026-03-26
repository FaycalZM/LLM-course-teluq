---
title: "7.1 Pipeline classique de clustering "
weight: 2
---

## L'art d'organiser le chaos sans professeur
Mes chers étudiants, nous entrons dans le royaume de l'**apprentissage non supervisé**.

Contrairement à la classification que nous avons vue en [**Semaine 4**]({{< relref "section-4-4.md" >}}), où nous disions à BERT : "Ceci est une critique positive", ici nous ne donnons aucune consigne. Nous disons simplement à la machine : "Regroupe ce qui se ressemble".

Pour y parvenir, nous ne pouvons pas simplement lancer un algorithme au hasard sur nos textes. Nous devons suivre un pipeline rigoureux en trois étapes, illustré par les **Figures 7-1 à 7-4 : Pipeline de clustering**. Ce pipeline est devenu le standard de l'industrie car il résout un problème mathématique majeur : **la malédiction de la dimensionnalité** ("*Curse of Dimensionality*").


## Étape 1 : La transformation vectorielle (Embedding)
La première brique, c'est l'**Embedding**, que nous avons déjà largement pratiqué. Comme le montre la **Figure 7-1**, nous transformons nos documents bruts (ici, des abstracts d'ArXiv) en vecteurs de nombres réels.

{{< bookfig src="112.png" week="07" >}}

> [!IMPORTANT]
🔑 **Je dois insister :** la qualité de votre clustering dépend à 80 % de la qualité de vos embeddings. 

> Si vous utilisez un modèle qui ne comprend pas bien votre domaine, les "points" dans l'espace seront mal placés, et vos groupes n'auront aucun sens sémantique. Pour nos travaux, nous privilégions des modèles comme `all-mpnet-base-v2` ou `thenlper/gte-small` car ils créent des sphères sémantiques très nettes.


## Étape 2 : Vaincre la malédiction de la dimensionnalité (UMAP)
C'est ici que les choses deviennent techniques et passionnantes. Nos vecteurs ont souvent 768 dimensions. Or, dans un espace à 768 dimensions, tous les points finissent par paraître "loin" les uns des autres (c'est la "malédiction"). De plus, la plupart des algorithmes de groupement s'essoufflent face à une telle complexité.

Nous devons réduire ces 768 dimensions à un petit nombre (généralement 5 ou 10) tout en préservant la structure du texte. Pour cela, nous utilisons **UMAP** (*Uniform Manifold Approximation and Projection*). 

Regardez la **Figure 7-2 : Réduction de dimensionnalité**. Imaginez que vos données soient une nappe froissée en boule dans une pièce (3D). UMAP est l'algorithme qui va essayer d'étaler la nappe sur le sol (2D) en faisant tout pour que deux points qui se touchaient dans la boule restent proches sur le sol.

{{< bookfig src="113.png" week="07" >}}

*   **Pourquoi UMAP et pas PCA ?** 
> [!WARNING]
⚠️ **Attention : erreur fréquente ici !** La PCA (Principal Component Analysis) est une méthode linéaire. Elle est excellente pour les chiffres simples, mais elle "écrase" les nuances subtiles du langage humain qui est fondamentalement non-linéaire. 

> UMAP, lui, préserve ce qu'on appelle la **structure locale** : il est obsédé par le fait de garder les voisins proches ensemble.

> [!NOTE]
🔑 **Note technique :** Lors de la configuration de UMAP, vous jouerez avec deux leviers :
> 1.  `n_neighbors` : définit si vous voulez une vision globale ou très locale.
> 2.  `min_dist` : définit à quel point les points peuvent être serrés les uns contre les autres.

{{< bookfig src="114.png" week="07" >}}


## Étape 3 : Le groupement intelligent (HDBSCAN)
Une fois nos données projetées dans un espace plus simple, il faut enfin créer les groupes. Oubliez le célèbre *K-Means* pour cette tâche. Le K-Means est comme un berger autoritaire qui force chaque mouton à appartenir à l'un des K enclos, même si certains moutons sont des loups ou des chèvres égarées.

{{< bookfig src="115.png" week="07" >}}


Nous utilisons **HDBSCAN** (*Hierarchical Density-Based Spatial Clustering of Applications with Noise*). Regardez la **Figure 7-5 : Centroid-based vs Density-based**.

{{< bookfig src="116.png" week="07" >}}

> *   **La force de la densité** : HDBSCAN ne cherche pas le "centre" d'un groupe. Il cherche les zones où les points sont très serrés (haute densité).
> *   **La sagesse du bruit** : C'est le point crucial. HDBSCAN a le droit de dire : "Ce document ne ressemble à rien d'autre, je le classe comme **bruit** (outlier)". Dans la **Figure 7-6**, vous verrez souvent des points gris : ce sont les documents que le modèle a refusé de grouper de force. C'est ce qui garantit que vos clusters finaux seront d'une pureté sémantique exceptionnelle.

{{< bookfig src="117.png" week="07" >}}


## Visualisation : La carte du savoir (Figure 7-6)
Une fois le pipeline exécuté, nous obtenons une visualisation comme celle de la **Figure 7-6 : Visualisation 2D des clusters**. Chaque couleur représente un thème découvert. 

> [!WARNING]
⚠️ Ne vous laissez pas séduire par les jolies couleurs.

> Une belle carte ne signifie pas un bon clustering. Vous devez toujours inspecter manuellement quelques documents dans chaque cluster pour valider que la machine a bien regroupé des concepts cohérents (ex: un groupe sur la traduction, un autre sur les modèles de langage médicaux, ...).


## Laboratoire de code : Implémentation du pipeline classique
Voici comment mettre en place ce flux de travail sur Google Colab avec le GPU T4. Nous allons utiliser un petit échantillon d'ArXiv pour comprendre la mécanique.

```python
# Installation des outils de pointe
# !pip install sentence-transformers umap-learn hdbscan pandas matplotlib arxiv

from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
import pandas as pd
import matplotlib.pyplot as plt
import arxiv  

# 1. RÉCUPÉRATION DES VRAIS RÉSUMÉS ARXIV
def fetch_arxiv_abstracts(query="machine learning", max_results=1000, categories=None):
    """
    Récupère les vrais résumés d'arXiv en utilisant la bibliothèque officielle arxiv.
    
    Paramètres :
    -----------
    query : str
        Requête de recherche (ex : "machine learning", "quantum computing", "nlp")
    max_results : int
        Nombre d'articles à récupérer (max 30 000 par limite API)
    categories : list
        Catégories arXiv spécifiques comme ['cs.LG', 'cs.CL', 'cs.AI']
    
    Retourne :
    --------
    list : Liste de chaînes de résumés
    """
    
    # Construction de la requête de recherche avec catégories si spécifiées
    if categories:
        cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
        full_query = f"({query}) AND ({cat_query})"
    else:
        full_query = query
    
    # Création de l'objet de recherche
    search = arxiv.Search(
        query=full_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    # Utilisation du Client pour une meilleure gestion des limites de taux
    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)
    
    abstracts = []
    titles = []
    ids = []
    
    print(f"Récupération de jusqu'à {max_results} articles pour la requête : '{query}'...")
    
    for result in client.results(search):
        abstracts.append(result.summary)
        titles.append(result.title)
        ids.append(result.entry_id.split('/')[-1])  # Extraction de l'ID arXiv
        
        if len(abstracts) % 100 == 0:
            print(f"  ... {len(abstracts)} résumés récupérés")
    
    print(f"Récupération réussie de {len(abstracts)} résumés")
    return abstracts, titles, ids

# --- RÉCUPÉRATION DES DONNÉES ---
# Exemple : Obtenir 1000 articles récents d'apprentissage automatique de cs.LG et cs.AI
docs, titles, ids = fetch_arxiv_abstracts(
    query="machine learning",
    max_results=1000,
    categories=["cs.LG", "cs.AI", "cs.CL"]  # Apprentissage automatique, IA, Langage et calcul
)


# 2. EMBEDDINGS ET CLUSTERING (Votre Code Original)
print("\nGénération des embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(docs, show_progress_bar=True)

print("Réduction de dimension avec UMAP...")
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
reduced_embeddings = umap_model.fit_transform(embeddings)

print("Clustering avec HDBSCAN...")
cluster_model = HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom')
clusters = cluster_model.fit_predict(reduced_embeddings)

# 3. VISUALISATION ET ANALYSE AMÉLIORÉES
# Création d'une projection 2D pour la visualisation
umap_2d = UMAP(n_neighbors=15, n_components=2, min_dist=0.1, metric='cosine', random_state=42)
embeddings_2d = umap_2d.fit_transform(embeddings)

# Tracé
plt.figure(figsize=(12, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                   c=clusters, cmap='Spectral', s=10, alpha=0.6)
plt.colorbar(scatter, label='ID du cluster (-1 = bruit)')
plt.title(f"Clustering de sujets arXiv : {len(set(clusters)) - (1 if -1 in clusters else 0)} clusters à partir de {len(docs)} articles")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.tight_layout()
plt.show()

```

## Éthique et Responsabilité : Les dangers de l'invisibilisation

> [!CAUTION]
⚠️ Mes chers étudiants, le clustering n'est pas un acte neutre. 
Lorsque vous utilisez HDBSCAN, l'algorithme rejette une partie des données comme étant du "bruit" (les outliers). 
1.  **Le risque de silence** : Si vous analysez des retours clients, les opinions très minoritaires ou les problèmes nouveaux risquent d'être classés en "bruit". Vous pourriez passer à côté d'un signal faible crucial (ex: une nouvelle faille de sécurité ou une plainte de discrimination) simplement parce qu'elle n'est pas assez "dense" statistiquement.
2.  **Biais de projection** : UMAP, en écrasant les dimensions, peut créer des proximités artificielles entre deux sujets qui n'ont rien à voir, ou au contraire séparer violemment deux concepts liés. 

> [!IMPORTANT]
🔑 **Mon conseil** : Ne considérez jamais le cluster "Outliers" (souvent noté -1) comme des déchets. C'est souvent là que se cachent les données les plus intéressantes, celles qui ne rentrent pas dans les cases. Explorez-les toujours avant de conclure votre analyse!

---
Vous maîtrisez maintenant l'architecture de base de la découverte thématique. Vous savez transformer une montagne de texte en une carte organisée. Mais il manque une chose : ces clusters n'ont pas encore de noms. Ils sont juste des "groupes de points". Dans la section suivante ➡️, nous allons découvrir **BERTopic**, l'outil qui va donner une voix à ces groupes grâce à la puissance des LLM.