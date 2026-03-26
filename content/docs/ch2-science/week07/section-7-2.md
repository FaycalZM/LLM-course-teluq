---
title: "7.2 Introduction à BERTopic"
weight: 3
---

## Au-delà du simple groupement : Donner une identité aux clusters
Bonjour à toutes et à tous ! J'espère que vous avez encore de l'énergie, car nous allons maintenant passer à la vitesse supérieure. Dans la section précédente, nous avons appris à regrouper des points colorés dans un espace. C'est une prouesse mathématique, certes, mais pour un utilisateur final, un "nuage de points bleu" ne veut rien dire. 
> [!IMPORTANT]
🔑 **Je dois insister :** l'objectif final du scientifique des données n'est pas seulement de grouper, mais d'**expliquer**. 

Aujourd'hui, nous allons découvrir **BERTopic**, le cadre de travail révolutionnaire créé par *Maarten Grootendorst*. C'est l'outil qui va permettre à vos clusters de "parler" et de nous dire exactement de quoi ils traitent.


## Qu'est-ce que BERTopic ? La philosophie des briques Lego
BERTopic n'est pas un simple algorithme, c'est une architecture modulaire. Comme l'illustre la **Figure 7-7 : La modularité de BERTopic**, vous devez imaginer ce système comme un assemblage de blocs Lego interchangeables. Si une nouvelle technique d'embedding sort demain, vous pouvez l'intégrer. Si vous préférez un autre algorithme de réduction, vous le changez.

{{< bookfig src="126.png" week="07" >}}


> [!NOTE]
🔑 **La distinction majeure avec les anciennes méthodes (comme LDA) :** Les approches traditionnelles comme la *Latent Dirichlet Allocation* (LDA) essayaient de deviner les sujets en regardant uniquement les fréquences de mots, sans rien comprendre au sens. 

> BERTopic, lui, commence par la compréhension (les embeddings Transformers) et finit par la statistique. C'est cette alliance du neuronal et du statistique qui le rend si puissant.


## L'architecture en 5 étapes
Regardons attentivement la **Figure 7-8 : Le pipeline de BERTopic** . Pour transformer des documents en sujets étiquetés, le modèle suit un cheminement précis :

{{< bookfig src="120.png" week="07" >}}

1.  **Embeddings** : On transforme les textes en vecteurs (SBERT). On capture la sémantique.
2.  **Réduction de dimension** : On utilise UMAP pour simplifier l'espace vectoriel tout en gardant les voisins ensemble.
3.  **Clustering** : On utilise HDBSCAN pour identifier les zones de haute densité sémantique.
4.  **Tokenisation** : On revient au texte ! On prend tous les documents d'un même cluster et on les traite comme un seul grand document.
5.  **Pondération c-TF-IDF** : On calcule quels mots sont les plus représentatifs de ce groupe précis.


<a id="c-tf-idf"></a>
## Le secret de fabrication : Le c-TF-IDF
C'est ici que réside le génie de BERTopic. Vous connaissez le TF-IDF classique ([**section 1.1**]({{< relref "section-1-1.md" >}}#tf-idf)). Mais ici, nous utilisons le **class-based TF-IDF (c-TF-IDF)**.

Observez la **Figure 7-9 : Génération du c-TF**.

{{< bookfig src="122.png" week="07" >}}

*   **Intuition** : Imaginez que vous ayez trois clusters : un sur la "Cuisine", un sur la "Politique", et un sur le "Sport". Le mot "balle" va apparaître très souvent dans le cluster "Sport". 
*   **Calcul** : Le c-TF-IDF mesure la fréquence d'un mot dans un cluster par rapport à sa fréquence dans tous les autres clusters réunis. Si "balle" est partout dans le sport mais nulle part ailleurs, il reçoit un score immense. 
*   **Résultat** : Cela nous donne une liste de mots-clés qui définissent l'identité unique de chaque groupe.

Comme le montre la **Figure 7-10 : Schéma de pondération**, cette formule mathématique agit comme un filtre qui élimine les mots banals pour ne laisser que l'essence thématique.

{{< bookfig src="123.png" week="07" >}}


## Exploration visuelle et interprétation

> [!WARNING]
⚠️ Un modèle que l'on ne peut pas explorer est un modèle inutile.

> BERTopic brille par ses capacités de visualisation, qui vous permettent de valider vos résultats.

### 1. La carte interactive des documents
La **Figure 7-11 : Sortie de la visualisation des documents** montre une projection 2D où chaque point est un document.

{{< bookfig src="127.png" week="07" >}}

*   **Analyse** : En survolant les points, vous voyez les titres des articles. Cela permet de vérifier visuellement si les clusters sont bien séparés ou s'ils se chevauchent. Un grand vide entre deux nuages de points indique des sujets très distincts (ex: Musique vs Physique Quantique). 
*   **Utilité** : C'est l'outil parfait pour repérer les erreurs de découpage ou les documents "égarés" au milieu d'un thème qui n'est pas le leur.

### 2. La hiérarchie des sujets
Souvent, vous aurez trop de sujets (par exemple 150). BERTopic permet de créer une structure hiérarchique. Le modèle peut fusionner les petits sujets similaires pour créer des "super-sujets". C'est comme passer d'une carte de ville à une carte de pays.

### 3. La matrice de corrélation (Heatmap)
Cette visualisation permet de voir quels sujets sont "cousins". Si vous avez un sujet sur les "Voitures électriques" et un autre sur les "Batteries au lithium", le modèle montrera une forte corrélation entre les deux, même s'il les a séparés au début.

## Laboratoire de code : BERTopic en action
Voici comment lancer une analyse thématique complète sur Colab. Notez la simplicité de l'interface malgré la complexité interne.

> [!NOTE]
Ce code n'est pas complèt. Voir le notebook Colab pour le pipeline complèt. 

```python
# Installation : !pip install bertopic sentence-transformers umap-learn hdbscan pandas matplotlib
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

# 1. Préparation des sous-composants (Modulaire !)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

# 2. Création du modèle BERTopic
topic_model = BERTopic(
  embedding_model=embedding_model,    # Compréhension
  umap_model=umap_model,              # Réduction
  hdbscan_model=hdbscan_model,        # Groupement
  verbose=True
)

# 3. Entraînement et extraction
# 'docs' est votre liste de textes
topics, probs = topic_model.fit_transform(docs)

# 4. Affichage des résultats
# Affiche les 10 thèmes les plus fréquents avec leurs mots-clés c-TF-IDF
print(topic_model.get_topic_info().head(10))

# 5. Visualisation interactive (nécessite un environnement notebook)
# fig = topic_model.visualize_topics()
# fig.show()
```

> [!WARNING]
⚠️ **Attention : erreur fréquente ici !** BERTopic utilise par défaut des modèles de `sentence-transformers`. Si vous avez un GPU (T4 sur Colab), assurez-vous que le modèle est bien chargé sur `cuda`. Cela divisera le temps de calcul par dix.

## Variantes et flexibilité : Quand adapter BERTopic ?
Comme je vous l'ai dit, BERTopic est une boîte à outils. Le livre mentionne plusieurs variantes passionnantes:

*   **Guided Topic Modeling** : Vous donnez une liste de mots-clés que vous *savez* être importants (ex: "COVID-19", "Vaccin"), et le modèle va orienter ses découvertes autour de ces ancres.
*   **Zero-shot Topic Modeling** : Vous définissez les sujets à l'avance et le modèle classe les documents, un peu comme ce que nous avons fait en [**section 4.4**]({{< relref "section-4-4.md" >}}#zero-shot), mais à l'échelle d'un modèle de sujet complet.
*   **Hierarchical Topic Modeling** : Pour naviguer du général au particulier.


## Éthique et Transparence : L'illusion du mot-clé

> [!CAUTION]
⚠️ Mes chers étudiants, les mots-clés peuvent être menteurs. 

Le c-TF-IDF vous donne les mots les plus fréquents d'un groupe, mais il ne vous donne pas le "ton" ou l'"intention". 
1.  **Réduction sémantique** : Un sujet étiqueté "Femme, Travail, Salaire" pourrait traiter de l'égalité salariale OU de préjugés sexistes. Les mots-clés ne vous disent pas si le propos est positif ou toxique.
2.  **Biais de fréquence** : Les mots très techniques ou rares, même s'ils sont cruciaux pour le sens, peuvent être écrasés par des mots plus communs s'ils ne sont pas assez fréquents dans le cluster. 


> [!IMPORTANT]
🔑 **Je dois insister :** Ne publiez jamais les résultats d'un BERTopic sans avoir lu au moins 5 à 10 documents par cluster. L'IA propose, l'humain dispose. Vous êtes les garde-fous de la cohérence sémantique.

---
Vous avez maintenant entre les mains l'un des outils les plus puissants de l'IA moderne. Vous ne voyez plus seulement des documents, vous voyez des structures. Dans la section suivante ➡️, nous verrons comment perfectionner encore ces thématiques en utilisant des techniques avancées pour affiner la représentation des mots : nous parlerons de **MMR** et de **reranking**.
