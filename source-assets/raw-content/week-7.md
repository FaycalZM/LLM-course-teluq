[CONTENU SEMAINE 7]

# Semaine 7 : Clustering et modélisation de sujets

**Titre : Découvrir la structure cachée : Clustering et BERTopic**

**Accroche du Professeur Khadidja Henni** : 
« Bonjour à toutes et à tous ! Je suis ravie de vous retrouver pour cette septième étape. La semaine dernière, nous avons appris à chercher une aiguille dans une botte de foin grâce à la recherche sémantique. Aujourd'hui, nous changeons de perspective : nous n'allons plus chercher une information précise, mais nous allons demander à la machine de nous décrire la forme de la botte de foin elle-même. 🔑 **Je dois insister :** le clustering est l'outil ultime de l'explorateur de données. C'est le passage de la lecture linéaire à la vision panoramique. Imaginez pouvoir cartographier 50 000 articles de recherche en un seul clic. C'est ce que nous allons accomplir ensemble. » [SOURCE: Livre p.137]

**Rappel semaine précédente** : « La semaine dernière, nous avons exploré la recherche sémantique dense, en apprenant à transformer des requêtes et des documents en vecteurs pour calculer leur proximité via la similarité cosinus et l'indexation FAISS. » [SOURCE: Detailed-plan.md]

**Objectifs de la semaine** :
À la fin de cette semaine, vous saurez :
*   Maîtriser le pipeline classique du clustering textuel (Embedding -> Réduction -> Groupement).
*   Comprendre mathématiquement l'intérêt de UMAP et HDBSCAN par rapport aux méthodes classiques.
*   Implémenter et configurer BERTopic pour extraire des thématiques cohérentes.
*   Utiliser des LLM pour générer des étiquettes de sujets compréhensibles par l'humain.
*   Interpréter et visualiser des nuages de documents complexes.

---

## 7.1 Pipeline classique de clustering (1300+ mots)

### L'art d'organiser le chaos sans professeur
« Mes chers étudiants, nous entrons dans le royaume de l'**apprentissage non supervisé**. » Contrairement à la classification que nous avons vue en Semaine 4, où nous disions à BERT : "Ceci est une critique positive", ici nous ne donnons aucune consigne. Nous disons simplement à la machine : "Regroupe ce qui se ressemble".

Pour y parvenir, nous ne pouvons pas simplement lancer un algorithme au hasard sur nos textes. Nous devons suivre un pipeline rigoureux en trois étapes, illustré par les **Figures 5-3 à 5-6 : Pipeline de clustering** (p.139-142 du livre). Ce pipeline est devenu le standard de l'industrie car il résout un problème mathématique majeur : la malédiction de la dimensionnalité. [SOURCE: Livre p.139]

### Étape 1 : La transformation vectorielle (Embedding)
La première brique, c'est l'**Embedding**, que nous avons déjà largement pratiqué. Comme le montre la **Figure 5-3** (p.139), nous transformons nos documents bruts (ici, des abstracts d'ArXiv) en vecteurs de nombres réels. 

🔑 **Je dois insister :** la qualité de votre clustering dépend à 80 % de la qualité de vos embeddings. Si vous utilisez un modèle qui ne comprend pas bien votre domaine, les "points" dans l'espace seront mal placés, et vos groupes n'auront aucun sens sémantique. Pour nos travaux, nous privilégions des modèles comme `all-mpnet-base-v2` ou `thenlper/gte-small` car ils créent des sphères sémantiques très nettes. [SOURCE: Livre p.140]

### Étape 2 : Vaincre la malédiction de la dimensionnalité (UMAP)
C'est ici que les choses deviennent techniques et passionnantes. Nos vecteurs ont souvent 768 dimensions. Or, dans un espace à 768 dimensions, tous les points finissent par paraître "loin" les uns des autres (c'est la "malédiction"). De plus, la plupart des algorithmes de groupement s'essoufflent face à une telle complexité.

Nous devons réduire ces 768 dimensions à un petit nombre (généralement 5 ou 10) tout en préservant la structure du texte. Pour cela, nous utilisons **UMAP** (*Uniform Manifold Approximation and Projection*). 

Regardez la **Figure 5-4 : Réduction de dimensionnalité** (p.141). Imaginez que vos données soient une nappe froissée en boule dans une pièce (3D). UMAP est l'algorithme qui va essayer d'étaler la nappe sur le sol (2D) en faisant tout pour que deux points qui se touchaient dans la boule restent proches sur le sol. 
*   **Pourquoi UMAP et pas PCA ?** ⚠️ **Attention : erreur fréquente ici !** La PCA (Principal Component Analysis) est une méthode linéaire. Elle est excellente pour les chiffres simples, mais elle "écrase" les nuances subtiles du langage humain qui est fondamentalement non-linéaire. UMAP, lui, préserve ce qu'on appelle la **structure locale** : il est obsédé par le fait de garder les voisins proches ensemble. [SOURCE: Livre p.141, Guide UMAP]

🔑 **Note technique :** Lors de la configuration de UMAP, vous jouerez avec deux leviers :
1.  `n_neighbors` : définit si vous voulez une vision globale ou très locale.
2.  `min_dist` : définit à quel point les points peuvent être serrés les uns contre les autres.

### Étape 3 : Le groupement intelligent (HDBSCAN)
Une fois nos données projetées dans un espace plus simple, il faut enfin créer les groupes. Oubliez le célèbre *K-Means* pour cette tâche. Le K-Means est comme un berger autoritaire qui force chaque mouton à appartenir à l'un des K enclos, même si certains moutons sont des loups ou des chèvres égarées.

Nous utilisons **HDBSCAN** (*Hierarchical Density-Based Spatial Clustering of Applications with Noise*). Regardez la **Figure 5-7 : Centroid-based vs Density-based** (p.143). 
*   **La force de la densité** : HDBSCAN ne cherche pas le "centre" d'un groupe. Il cherche les zones où les points sont très serrés (haute densité).
*   **La sagesse du bruit** : C'est le point crucial. HDBSCAN a le droit de dire : "Ce document ne ressemble à rien d'autre, je le classe comme **bruit** (outlier)". Dans la **Figure 5-8** (p.145), vous verrez souvent des points gris : ce sont les documents que le modèle a refusé de grouper de force. C'est ce qui garantit que vos clusters finaux seront d'une pureté sémantique exceptionnelle. [SOURCE: Livre p.142-143, Figure 5-7]

### Visualisation : La carte du savoir (Figure 5-8)
Une fois le pipeline exécuté, nous obtenons une visualisation comme celle de la **Figure 5-8 : Visualisation 2D des clusters** (p.145). Chaque couleur représente un thème découvert. 
⚠️ **Fermeté bienveillante** : « Ne vous laissez pas séduire par les jolies couleurs. » Une belle carte ne signifie pas un bon clustering. Vous devez toujours inspecter manuellement quelques documents dans chaque cluster pour valider que la machine a bien regroupé des concepts cohérents (ex: un groupe sur la traduction, un autre sur les modèles de langage médicaux). [SOURCE: Livre p.145]

### Laboratoire de code : Implémentation du pipeline classique
Voici comment mettre en place ce flux de travail sur Google Colab avec le GPU T4. Nous allons utiliser un petit échantillon d'ArXiv pour comprendre la mécanique.

```python
# Installation des outils de pointe
# !pip install sentence-transformers umap-learn hdbscan pandas matplotlib

from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
import pandas as pd
import matplotlib.pyplot as plt

# 1. Chargement des données (QUESTION CODE)
# Imaginons une liste de 1000 abstracts d'ArXiv
docs = ["Abstract 1...", "Abstract 2...", "..."] # Chargé via datasets

# --- RÉPONSE COMPLÈTE (ANSWER CODE) ---
# [SOURCE: Pipeline classique Livre p.139-142]

# ÉTAPE A : Embeddings denses
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(docs, show_progress_bar=True)

# ÉTAPE B : Réduction de dimension avec UMAP
# On réduit à 5 dimensions pour aider le clustering, puis à 2 pour le graphique
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
reduced_embeddings = umap_model.fit_transform(embeddings)

# ÉTAPE C : Clustering avec HDBSCAN
# min_cluster_size=10 : il faut au moins 10 documents pour faire un thème
cluster_model = HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom')
clusters = cluster_model.fit_predict(reduced_embeddings)

# Visualisation rapide
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=clusters, cmap='Spectral', s=5)
plt.title("Visualisation des thématiques découvertes")
plt.show()
```

### Éthique et Responsabilité : Les dangers de l'invisibilisation
⚠️ **Éthique ancrée** : « Mes chers étudiants, le clustering n'est pas un acte neutre. » 
Lorsque vous utilisez HDBSCAN, l'algorithme rejette une partie des données comme étant du "bruit" (les outliers). 
1.  **Le risque de silence** : Si vous analysez des retours clients, les opinions très minoritaires ou les problèmes nouveaux risquent d'être classés en "bruit". Vous pourriez passer à côté d'un signal faible crucial (ex: une nouvelle faille de sécurité ou une plainte de discrimination) simplement parce qu'elle n'est pas assez "dense" statistiquement.
2.  **Biais de projection** : UMAP, en écrasant les dimensions, peut créer des proximités artificielles entre deux sujets qui n'ont rien à voir, ou au contraire séparer violemment deux concepts liés. 

🔑 **Mon conseil de professeur** : Ne considérez jamais le cluster "Outliers" (souvent noté -1) comme des déchets. C'est souvent là que se cachent les données les plus intéressantes, celles qui ne rentrent pas dans les cases. Explorez-les toujours avant de conclure votre analyse. [SOURCE: Livre p.28, p.144]

« Vous maîtrisez maintenant l'architecture de base de la découverte thématique. Vous savez transformer une montagne de texte en une carte organisée. Mais il manque une chose : ces clusters n'ont pas encore de noms. Ils sont juste des "groupes de points". Dans la section suivante, nous allons découvrir **BERTopic**, l'outil qui va donner une voix à ces groupes grâce à la puissance des LLM. »

---
*Fin de la section 7.1 (1340 mots environ)*
## 7.2 Introduction à BERTopic (1400+ mots)

### Au-delà du simple groupement : Donner une identité aux clusters
« Bonjour à toutes et à tous ! J'espère que vous avez encore de l'énergie, car nous allons maintenant passer à la vitesse supérieure. Dans la section précédente, nous avons appris à regrouper des points colorés dans un espace. C'est une prouesse mathématique, certes, mais pour un utilisateur final, un "nuage de points bleu" ne veut rien dire. 🔑 **Je dois insister :** l'objectif final du scientifique des données n'est pas seulement de grouper, mais d'**expliquer**. Aujourd'hui, nous allons découvrir **BERTopic**, le cadre de travail révolutionnaire créé par Maarten Grootendorst. C'est l'outil qui va permettre à vos clusters de "parler" et de nous dire exactement de quoi ils traitent. » [SOURCE: Livre p.148]

### Qu'est-ce que BERTopic ? La philosophie des briques Lego
BERTopic n'est pas un simple algorithme, c'est une architecture modulaire. Comme l'illustre la **Figure 5-17 : La modularité de BERTopic** (p.151 du livre), vous devez imaginer ce système comme un assemblage de blocs Lego interchangeables. Si une nouvelle technique d'embedding sort demain, vous pouvez l'intégrer. Si vous préférez un autre algorithme de réduction, vous le changez.

🔑 **La distinction majeure avec les anciennes méthodes (comme LDA) :** Les approches traditionnelles comme la *Latent Dirichlet Allocation* (LDA) essayaient de deviner les sujets en regardant uniquement les fréquences de mots, sans rien comprendre au sens. BERTopic, lui, commence par la compréhension (les embeddings Transformers) et finit par la statistique. C'est cette alliance du neuronal et du statistique qui le rend si puissant. [SOURCE: Livre p.148-151, Figure 5-17]

### L'architecture en 5 étapes (Figure 5-11)
Regardons attentivement la **Figure 5-11 : Le pipeline de BERTopic** (p.148). Pour transformer des documents en sujets étiquetés, le modèle suit un cheminement précis :

1.  **Embeddings** : On transforme les textes en vecteurs (SBERT). On capture la sémantique.
2.  **Réduction de dimension** : On utilise UMAP pour simplifier l'espace vectoriel tout en gardant les voisins ensemble.
3.  **Clustering** : On utilise HDBSCAN pour identifier les zones de haute densité sémantique.
4.  **Tokenisation** : On revient au texte ! On prend tous les documents d'un même cluster et on les traite comme un seul grand document.
5.  **Pondération c-TF-IDF** : On calcule quels mots sont les plus représentatifs de ce groupe précis. [SOURCE: Livre p.148, Figure 5-11]

### Le secret de fabrication : Le c-TF-IDF
C'est ici que réside le génie de BERTopic. Vous connaissez le TF-IDF classique (section 1.1). Mais ici, nous utilisons le **class-based TF-IDF (c-TF-IDF)**.

Observez la **Figure 5-13 : Génération du c-TF** (p.149). 
*   **Intuition** : Imaginez que vous ayez trois clusters : un sur la "Cuisine", un sur la "Politique", et un sur le "Sport". Le mot "balle" va apparaître très souvent dans le cluster "Sport". 
*   **Calcul** : Le c-TF-IDF mesure la fréquence d'un mot dans un cluster par rapport à sa fréquence dans tous les autres clusters réunis. Si "balle" est partout dans le sport mais nulle part ailleurs, il reçoit un score immense. 
*   **Résultat** : Cela nous donne une liste de mots-clés qui définissent l'identité unique de chaque groupe. Comme le montre la **Figure 5-14 : Schéma de pondération** (p.150), cette formule mathématique agit comme un filtre qui élimine les mots banals pour ne laisser que l'essence thématique. [SOURCE: Livre p.149-150, Figures 5-13, 5-14]

### Exploration visuelle et interprétation
⚠️ **Fermeté bienveillante** : « Un modèle que l'on ne peut pas explorer est un modèle inutile. » BERTopic brille par ses capacités de visualisation, qui vous permettent de valider vos résultats.

#### 1. La carte interactive des documents (Figure 5-18)
La **Figure 5-18 : Sortie de la visualisation des documents** (p.155) montre une projection 2D où chaque point est un document. 
*   **Analyse** : En survolant les points, vous voyez les titres des articles. Cela permet de vérifier visuellement si les clusters sont bien séparés ou s'ils se chevauchent. Un grand vide entre deux nuages de points indique des sujets très distincts (ex: Musique vs Physique Quantique). 
*   **Utilité** : C'est l'outil parfait pour repérer les erreurs de découpage ou les documents "égarés" au milieu d'un thème qui n'est pas le leur. [SOURCE: Livre p.155, Figure 5-18]

#### 2. La hiérarchie des sujets
Souvent, vous aurez trop de sujets (par exemple 150). BERTopic permet de créer une structure hiérarchique. Le modèle peut fusionner les petits sujets similaires pour créer des "super-sujets". C'est comme passer d'une carte de ville à une carte de pays.

#### 3. La matrice de corrélation (Heatmap)
Cette visualisation permet de voir quels sujets sont "cousins". Si vous avez un sujet sur les "Voitures électriques" et un autre sur les "Batteries au lithium", le modèle montrera une forte corrélation entre les deux, même s'il les a séparés au début.

### Laboratoire de code : BERTopic en action
Voici comment lancer une analyse thématique complète sur Colab. Notez la simplicité de l'interface malgré la complexité interne.

```python
# Installation : !pip install bertopic
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

# 1. Préparation des sous-composants (Modulaire !)
# [SOURCE: Choix des modèles recommandés p.140, p.152]
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

# 2. Création du modèle BERTopic
# [SOURCE: Architecture modulaire Figure 5-17]
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

⚠️ **Attention : erreur fréquente ici !** BERTopic utilise par défaut des modèles de `sentence-transformers`. Si vous avez un GPU (T4 sur Colab), assurez-vous que le modèle est bien chargé sur `cuda`. Cela divisera le temps de calcul par dix.

### Variantes et flexibilité : Quand adapter BERTopic ?
Comme je vous l'ai dit, BERTopic est une boîte à outils. Le livre mentionne plusieurs variantes passionnantes (p.152) :
*   **Guided Topic Modeling** : Vous donnez une liste de mots-clés que vous *savez* être importants (ex: "COVID-19", "Vaccin"), et le modèle va orienter ses découvertes autour de ces ancres.
*   **Zero-shot Topic Modeling** : Vous définissez les sujets à l'avance et le modèle classe les documents, un peu comme ce que nous avons fait en section 4.4, mais à l'échelle d'un modèle de sujet complet.
*   **Hierarchical Topic Modeling** : Pour naviguer du général au particulier. [SOURCE: Livre p.152, "The Modularity of BERTopic"]

### Éthique et Transparence : L'illusion du mot-clé
⚠️ **Éthique ancrée** : « Mes chers étudiants, les mots-clés peuvent être menteurs. » 
Le c-TF-IDF vous donne les mots les plus fréquents d'un groupe, mais il ne vous donne pas le "ton" ou l'"intention". 
1.  **Réduction sémantique** : Un sujet étiqueté "Femme, Travail, Salaire" pourrait traiter de l'égalité salariale OU de préjugés sexistes. Les mots-clés ne vous disent pas si le propos est positif ou toxique.
2.  **Biais de fréquence** : Les mots très techniques ou rares, même s'ils sont cruciaux pour le sens, peuvent être écrasés par des mots plus communs s'ils ne sont pas assez fréquents dans le cluster. 

🔑 **Je dois insister :** Ne publiez jamais les résultats d'un BERTopic sans avoir lu au moins 5 à 10 documents par cluster. L'IA propose, l'humain dispose. Vous êtes les garde-fous de la cohérence sémantique. [SOURCE: Livre p.28, p.144]

« Vous avez maintenant entre les mains l'un des outils les plus puissants de l'IA moderne. Vous ne voyez plus seulement des documents, vous voyez des structures. Dans la section suivante, nous verrons comment perfectionner encore ces thématiques en utilisant des techniques avancées pour affiner la représentation des mots : nous parlerons de MMR et de reranking. »

---
*Fin de la section 7.2 (1420 mots environ)*
## 7.3 Représentation des sujets (1600+ mots)

### L'art de sculpter l'identité des thématiques
« Bonjour à toutes et à tous ! J'espère que vous avez pris le temps d'admirer vos nuages de points de la section précédente. C'est un excellent début, mais je vais être honnête avec vous : un nuage de points sans une description précise, c'est comme une carte de géographie sans noms de villes. Jusqu'ici, nous avons laissé la statistique brute du c-TF-IDF (section 7.2) choisir les mots-clés. 🔑 **Je dois insister :** la statistique est puissante, mais elle est parfois aveugle aux subtilités sémantiques. Aujourd'hui, nous allons apprendre à "sculpter" les représentations de nos sujets pour qu'elles soient non seulement précises, mais aussi diversifiées et pertinentes. Nous allons passer de la simple liste de mots à une véritable signature thématique. » [SOURCE: Livre p.156]

### Le socle : c-TF-IDF et ses limites "sac de mots"
Avant de passer aux techniques avancées, comprenons bien notre point de départ. Le **c-TF-IDF** agit comme un premier filtre. Il identifie les mots qui apparaissent plus fréquemment dans un cluster spécifique que dans le reste du corpus. C'est une approche dite "Bag-of-words" (sac de mots). 

Le problème ? Le c-TF-IDF ne "comprend" pas le sens des mots. Si un mot comme "données" apparaît très souvent dans un cluster sur l'IA et aussi dans un cluster sur la biologie, il risque d'avoir un score faible, alors qu'il est crucial pour les deux. De plus, il peut laisser passer des variantes grammaticales inutiles (ex: "chat", "chats", "chaton") qui encombrent votre liste de mots-clés. C'est pour corriger ces défauts que nous introduisons le **Reranking** (ré-ordonnancement). [SOURCE: Livre p.156, Figure 5-19]

### Le concept de Reranking (Figure 5-19)
Regardons attentivement la **Figure 5-19 : Affiner les représentations de sujets** (p.156). 
*   **À gauche (Original topic)** : On voit une liste de mots générée par le c-TF-IDF pur. L'ordre est purement statistique. Certains mots en haut de liste peuvent être génériques.
*   **Le bloc central (Reranker)** : C'est le "cerveau" additionnel que nous ajoutons. Son rôle est de reprendre les candidats fournis par la statistique et de les passer au crible de la sémantique.
*   **À droite (Reranked topic)** : La liste est réorganisée. Les mots qui capturent le mieux l'essence sémantique du groupe montent en grade, tandis que le "bruit" statistique descend. 

🔑 **Notez bien cette intuition :** On ne change pas le contenu du sac, on change simplement l'ordre dans lequel on sort les objets du sac pour présenter les plus beaux en premier. [SOURCE: Livre p.156, Figure 5-19]

### KeyBERTInspired : Quand les embeddings jugent les mots
La première technique de pointe que nous étudions est **KeyBERTInspired**. Cette méthode est une adaptation de l'algorithme KeyBERT au monde du topic modeling. 

Regardez la **Figure 5-20 : Le bloc de reranking** (p.157). Elle illustre comment ce bloc vient s'enficher par-dessus la couche de représentation. 
1.  **Le Centroïde du sujet** : Pour chaque cluster, nous calculons la moyenne de tous les embeddings des documents qui le composent. C'est le "poids lourd" sémantique du sujet, sa position GPS idéale.
2.  **La comparaison** : Nous prenons les mots-clés candidats (générés par c-TF-IDF) et nous les transformons eux aussi en vecteurs.
3.  **Le calcul de similarité** : Nous calculons la similarité cosinus (vue en 6.2) entre le vecteur du mot et le centroïde du sujet.
4.  **Le verdict** : Si un mot a une fréquence élevée (statistique) ET qu'il est sémantiquement très proche du cœur du sujet (neuronal), il devient le candidat numéro 1. 

🔑 **Je dois insister :** KeyBERTInspired permet d'éliminer les mots qui sont là par "accident statistique" mais qui n'ont rien à voir avec le thème central. C'est un filtre de cohérence. [SOURCE: Livre p.157-158, Figure 5-20]

### Vaincre la redondance : Maximal Marginal Relevance (MMR)
⚠️ **Attention : erreur fréquente ici !** Beaucoup d'étudiants pensent qu'avoir les 10 mots les plus "proches" du sujet est la solution parfaite. Mais imaginez un sujet dont les mots-clés sont : "Espace", "Galaxie", "Cosmos", "Univers", "Spatial", "Céleste"... C'est redondant ! Vous avez utilisé six mots pour dire la même chose.

Pour résoudre cela, nous utilisons la **Maximal Marginal Relevance (MMR)**. Comme l'illustre la **Figure 5-21 : Empiler plusieurs blocs** (p.157), MMR est souvent la dernière brique du mur.
*   **Le principe** : MMR essaie de maximiser deux choses contradictoires : la **pertinence** par rapport au sujet et la **diversité** par rapport aux mots déjà choisis. 
*   **Le fonctionnement** : 
    1. On choisit le mot le plus pertinent. 
    2. Pour le deuxième mot, on cherche celui qui est pertinent MAIS qui est le plus "différent" (vecteur éloigné) du premier mot choisi. 
    3. On continue ainsi pour couvrir toutes les facettes du sujet.

🔑 **L'intuition du Professeur Henni :** MMR, c'est comme constituer une équipe de projet. Vous ne voulez pas 10 clones identiques du meilleur ingénieur ; vous voulez un ingénieur, un designer, un commercial et un juriste. Ils sont tous pertinents pour le projet, mais ils apportent des perspectives différentes. [SOURCE: Livre p.159, Figure 5-21]

### Mise à jour dynamique des représentations
L'un des avantages incroyables de BERTopic est que vous pouvez changer la façon dont vos sujets sont décrits *sans avoir à tout recalculer*. 

Imaginez : vous avez mis 2 heures à générer vos embeddings et à calculer vos clusters sur 100 000 documents. Vous vous rendez compte que les noms de sujets sont bof. ⚠️ **Fermeté bienveillante** : Ne relancez pas tout ! Utilisez la méthode `.update_topics()`. Elle ne touche pas à la structure des groupes (les points ne bougent pas), elle change seulement l'algorithme qui "étiquette" ces groupes. C'est quasi instantané. [SOURCE: Livre p.158]

### Laboratoire de code : Affiner les thématiques
Voici comment implémenter ce pipeline de précision sur Google Colab. Nous allons stacker (empiler) KeyBERT et MMR pour obtenir des résultats professionnels.

```python
# Installation requise : !pip install bertopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from bertopic import BERTopic

# 1. On définit nos modèles de sculpture sémantique
# [SOURCE: Paramètres recommandés p.159-160]
keybert_model = KeyBERTInspired()
mmr_model = MaximalMarginalRelevance(diversity=0.3) # 0.3 est un bon équilibre

# 2. On crée une liste de modèles de représentation
# BERTopic va les appliquer l'un après l'autre
representation_model = {
    "Main": [keybert_model, mmr_model]
}

# 3. Supposons que topic_model est déjà entraîné (vu en 7.2)
# On met à jour les représentations sans toucher aux clusters
# [SOURCE: Utilisation de update_topics p.158]
topic_model.update_topics(
    docs, 
    representation_model=representation_model
)

# 4. Comparaison
# Regardez la différence entre les mots-clés avant et après !
print(topic_model.get_topic_info()[["Topic", "Count", "Name", "Representation"]].head())
```

🔑 **Note technique sur le paramètre `diversity`** : Si vous le réglez à 0, MMR ne fait rien (on prend juste les plus proches). Si vous le réglez à 1, le modèle choisira des mots qui n'ont parfois aucun rapport entre eux juste pour être "différent". La valeur magique se situe souvent entre 0.2 et 0.5.

### Cas d'usage : Domaines techniques vs Domaines créatifs
Pourquoi ces réglages sont-ils vitaux ? 
*   **En Médecine (Figure 5-21 scenario)** : c-TF-IDF pourrait vous donner "Patient", "Hôpital", "Soin". KeyBERTInspired va forcer le modèle à regarder les termes cliniques précis comme "Cardiopathie" ou "Insuffisance". MMR va s'assurer que vous n'avez pas juste 10 synonymes du mot "douleur".
*   **En Analyse de Presse** : MMR est indispensable. Un sujet sur une élection pourrait être saturé par le nom du gagnant. MMR va forcer le modèle à inclure "vote", "partis", "campagne" et "sondages" pour donner une image complète de l'événement. [SOURCE: Livre p.157]

### Éthique et Responsabilité : Le pouvoir du cadrage (Framing)
⚠️ **Éthique ancrée** : « Mes chers étudiants, les mots que vous choisissez de montrer à vos clients ou à vos décideurs créent une réalité. » 
Lorsque vous utilisez MMR ou KeyBERT pour "nettoyer" vos sujets, vous faites un choix éditorial. 
1.  **La réduction du complexe** : En voulant des mots-clés "propres" et "uniques", vous risquez d'effacer les contradictions internes d'un sujet. Un groupe de documents peut être très divisé sur une question, mais MMR va lisser cela pour donner une image de diversité harmonieuse.
2.  **Biais de centroïde** : KeyBERTInspired se base sur la "moyenne" du cluster. Cela signifie que les opinions extrêmes ou les cas particuliers au sein d'un thème seront systématiquement écartés de la description. C'est une forme de "tyrannie de la majorité sémantique". 

🔑 **Mon conseil de professeur** : Utilisez ces techniques pour rendre vos résultats lisibles, mais gardez toujours la liste c-TF-IDF originale sous le coude. Elle est moins "jolie", mais elle est plus fidèle à la distribution brute des données. Ne sacrifiez jamais la vérité à l'esthétique. [SOURCE: Livre p.28, p.161]

« Vous avez maintenant appris à transformer des groupes de données en thématiques intelligentes, denses et diversifiées. Votre carte commence à avoir de l'allure ! Mais nous pouvons aller encore plus loin. Les mots-clés, c'est bien, mais une phrase complète écrite par un humain, c'est mieux. Dans la prochaine section, nous allons voir comment inviter des modèles comme GPT-4 à la table pour qu'ils deviennent les "rédacteurs en chef" de vos thématiques. »

---
*Fin de la section 7.3 (1620 mots environ)*
## 7.4 Amélioration par les LLM (1200+ mots)

### L'IA comme rédactrice en chef : Le problème du "Dernier Kilomètre"
« Bonjour à toutes et à tous ! J'espère que vous avez bien en tête nos signatures thématiques de la section précédente. Nous avons réussi à transformer des nuages de points en listes de mots-clés intelligents grâce à KeyBERT et MMR. C'est une victoire technique majeure. Mais posons-nous une question de "vérité terrain" : si vous présentez un rapport à votre direction avec comme titre de sujet : "Espace | Galaxie | Télescope | NASA", vont-ils comprendre l'essence du message ? Probablement. Mais si vous écrivez : "Avancées récentes de l'imagerie spatiale par le télescope James Webb", vous changez de dimension. 🔑 **Je dois insister :** les mots-clés sont des indices, mais la phrase est une information. Aujourd'hui, nous allons apprendre à utiliser les LLM génératifs comme la touche finale, le "vernis" qui va transformer vos données en connaissances exploitables par des humains. » [SOURCE: Livre p.160]

Ce que nous appelons le "problème du dernier kilomètre" en science des données, c'est cette difficulté à rendre un résultat mathématique parfaitement intelligible pour un non-expert. Jusqu'à l'arrivée des modèles génératifs, nous étions bloqués. Aujourd'hui, nous allons inviter GPT-4, Llama-3 ou Mistral à la table pour qu'ils deviennent les narrateurs de vos thématiques.

### L'intuition technique : Le LLM comme agrégateur de contexte
Comment un modèle peut-il "nommer" un groupe de 5 000 documents sans les lire un par un ? La réponse réside dans la sélection intelligente d'échantillons. BERTopic ne demande pas au LLM de traiter l'intégralité du cluster (ce qui serait ruineux en termes de jetons/tokens et de temps). 

Regardez attentivement la **Figure 5-23 : Utilisation des LLM pour la génération de labels** (p.161 du livre). Cette illustration détaille un processus en trois temps :
1.  **La sélection des délégués** : Pour chaque cluster, BERTopic identifie les documents les plus représentatifs (souvent les 3 ou 4 documents les plus proches du centre sémantique, le "centroïde"). 
2.  **La signature statistique** : On y ajoute les mots-clés c-TF-IDF que nous avons calculés en section 7.2.
3.  **Le Prompt de synthèse** : On envoie cet ensemble au LLM avec une consigne précise.

🔑 **L'astuce de performance du Prof. Henni :** En ne fournissant que les "délégués" du sujet, on permet au LLM de saisir le contexte profond sans saturer sa fenêtre de contexte. C'est une forme de compression sémantique ultra-efficace. [SOURCE: Livre p.161, Figure 5-23]

### L'art du Prompting pour la modélisation de sujets
⚠️ **Attention : erreur fréquente ici !** Si vous demandez simplement au LLM : "Donne un titre à ces mots", vous obtiendrez des résultats banals. Pour obtenir une étiquette de haute qualité, votre prompt doit être une architecture à part entière.

Un prompt de modélisation de sujets efficace doit contenir :
*   **Le rôle (Persona)** : "Vous êtes un expert en analyse documentaire et en taxonomie."
*   **Le contexte des données** : "Ces documents sont des résumés d'articles scientifiques provenant d'ArXiv."
*   **Les preuves (Documents représentatifs)** : "Voici trois exemples de textes appartenant à ce groupe..."
*   **Les indices (Mots-clés)** : "Les mots statistiquement dominants sont : [KEYWORDS]."
*   **La contrainte de sortie** : "Fournissez un titre court (moins de 7 mots) et une description d'une phrase."

🔑 **Je dois insister :** plus vous donnez de structure à votre demande, moins le modèle aura tendance à "halluciner" un titre qui n'a rien à voir avec vos données. [SOURCE: Livre p.162, "Prompt Engineering for Topics"]

### Modèles propriétaires (API) vs Modèles Open-source
Dans le cadre de BERTopic, vous avez deux grandes options pour cette étape finale.

#### 1. Les API de haut niveau (OpenAI, Anthropic, Cohere)
*   **Avantages** : Qualité de résumé exceptionnelle, gestion de l'ironie et des nuances complexes.
*   **Inconvénients** : Coût (facturation au token) et confidentialité (vos documents "délégués" sortent de votre infrastructure).
*   **Usage idéal** : Rapports stratégiques, analyse de presse internationale.

#### 2. Les modèles locaux (Flan-T5, Mistral, Llama, Phi-3)
C'est ici que votre GPU T4 de Colab brille. Comme nous ne traitons que quelques phrases par sujet, même un petit modèle comme **Flan-T5-base** (p.161) peut faire un travail remarquable.
*   **Flan-T5** : Un modèle de type Encodeur-Décodeur, excellent pour le résumé pur et simple.
*   **Phi-3 / Llama-3-8B** : Plus créatifs, ils permettent de générer des labels plus "humains" et moins robotiques. [SOURCE: Livre p.161-162]

### Laboratoire de code : Intégration LLM dans BERTopic
Voici comment configurer BERTopic pour utiliser un LLM comme "moteur d'étiquetage". Nous allons simuler l'usage d'un modèle local pour rester dans une approche respectueuse de vos ressources GPU.

```python
# Installation requise : !pip install bertopic transformers accelerate bitsandbytes
from bertopic import BERTopic
from bertopic.representation import TextGeneration
from transformers import pipeline

# 1. Préparation du "Cerveau" de synthèse (Modèle local Phi-3-mini)
# [SOURCE: Choix de modèle compact p.54, p.162]
generator = pipeline(
    "text-generation", 
    model="microsoft/Phi-3-mini-4k-instruct", 
    device_map="auto",
    model_kwargs={"torch_dtype": "auto", "trust_remote_code": True}
)

# 2. Construction du Prompt Template spécialisé
# [SOURCE: Anatomie d'un prompt Livre p.173-176]
prompt = """
I have a topic described by the following keywords: [KEYWORDS]
Based on the following example documents from this topic:
[DOCUMENTS]

Extract a concise, professional label for this topic.
Topic Label:"""

# 3. Création du bloc de représentation LLM
representation_model = TextGeneration(generator, prompt=prompt)

# 4. Intégration dans BERTopic (Supposons que topic_model est déjà créé)
# Nous utilisons update_topics pour ne pas perdre les calculs précédents
# [SOURCE: Méthode update_topics p.158]
topic_model.update_topics(
    docs, 
    representation_model=representation_model
)

# 5. Résultat : Vos sujets ont maintenant un label généré par l'IA !
print(topic_model.get_topic_info()[["Topic", "CustomName", "Representation"]].head())
```

### Évaluation des labels générés : Comment savoir si l'IA ment ?
⚠️ **Fermeté bienveillante** : « Ne tombez pas dans le piège de la beauté. » Un titre magnifique généré par GPT-4 peut être une hallucination complète si le modèle a mal interprété les documents délégués.
Comment vérifier ?
1.  **Cohérence intra-cluster** : Le titre correspond-il vraiment aux 10 premiers mots-clés du c-TF-IDF ?
2.  **Test de spécificité** : Si l'IA donne le même titre ("Informatique générale") à trois clusters différents, votre prompt n'est pas assez précis. 
3.  **L'audit humain** : 🔑 **C'est non-négociable :** vous devez lire les documents délégués et juger si le titre résume honnêtement leur contenu. [SOURCE: Livre p.163]

### Éthique et Responsabilité : La tentation du "Rebranding"
⚠️ **Éthique ancrée** : « Mes chers étudiants, nommer, c'est exercer un pouvoir. » 
Lorsque vous demandez à une IA de nommer un cluster de données sensibles (ex: des plaintes de citoyens ou des effets secondaires de médicaments) :
1.  **L'euphémisation** : L'IA, entraînée à être "polie" (RLHF), pourrait transformer un cluster de "Plaintes pour racisme" en "Défis de communication interculturelle". C'est un biais de neutralité dangereux qui masque la réalité sociale des données.
2.  **La stigmatisation** : À l'inverse, un prompt mal cadré pourrait pousser le modèle à utiliser des termes chargés d'émotion ou de préjugés pour décrire un groupe de population identifié par le clustering. 

🔑 **Mon conseil de professeur** : Considérez toujours le label généré par le LLM comme une **suggestion**, et non comme une vérité définitive. Dans un pipeline de production, prévoyez toujours une étape de validation humaine pour les noms de sujets avant qu'ils ne soient diffusés. L'IA est une excellente assistante de rédaction, mais elle ne doit jamais être la seule juge de la signification de vos données. [SOURCE: Livre p.28]

« Vous avez maintenant parcouru tout le chemin : de la donnée brute à la connaissance structurée et nommée. Vous avez appris à transformer une pile de papier en une carte interactive où chaque pays a un nom clair. C'est une compétence rare et précieuse. Place maintenant au laboratoire pour mettre en œuvre votre premier pipeline de cartographie sémantique complet ! »

---
*Fin de la section 7.4 (1260 mots environ)*
## 🧪 LABORATOIRE SEMAINE 7 (600+ mots)

**Accroche du Professeur Khadidja Henni** : 
« Bonjour à toutes et à tous ! Nous y sommes : le moment de transformer une montagne de documents illisibles en une carte thématique structurée. Dans ce laboratoire, nous allons mettre en pratique notre "vision panoramique". 🔑 **Je dois insister :** ne vous contentez pas de regarder les jolies bulles colorées. Un bon scientifique de la donnée est celui qui va fouiller dans les "outliers" (les points gris) pour comprendre ce que la machine n'a pas réussi à classer. Prêt·e·s à organiser le chaos ? C'est parti ! » [SOURCE: Livre p.137, p.144]

---

### 🔹 QUIZ MCQ (10 questions)

1. **Quel algorithme de clustering utilisé dans BERTopic permet de détecter automatiquement le nombre de clusters sans que l'humain ait à le fixer ?**
   a) K-Means
   b) PCA
   c) HDBSCAN
   d) Cosine Similarity
   **[Réponse: c]** [Explication: Contrairement à K-Means qui nécessite un nombre "K" défini, HDBSCAN groupe les points par densité et trouve seul le nombre de thèmes. SOURCE: Livre p.143, Figure 5-7]

2. **Quelle technique de réduction de dimensionnalité est recommandée pour le texte car elle préserve mieux les relations non-linéaires et les voisinages locaux ?**
   a) PCA (Principal Component Analysis)
   b) UMAP (Uniform Manifold Approximation and Projection)
   c) TF-IDF
   d) Word2Vec
   **[Réponse: b]** [Explication: UMAP est capable de "déplier" des structures sémantiques complexes là où la PCA, linéaire, échouerait. SOURCE: Livre p.141-142]

3. **Quel paramètre de HDBSCAN contrôle la taille minimale pour qu'un groupe de documents soit considéré comme un "sujet" et non comme du bruit ?**
   a) `n_neighbors`
   b) `min_dist`
   c) `min_cluster_size`
   d) `diversity`
   **[Réponse: c]** [Explication: Si vous fixez ce paramètre à 50, tout groupe de moins de 50 documents sera ignoré et classé en "outlier". SOURCE: Livre p.144]

4. **Qu'est-ce que le c-TF-IDF dans l'architecture BERTopic ?**
   a) Une méthode pour réduire la taille des vecteurs.
   b) Une variante du TF-IDF qui calcule l'importance d'un mot au sein d'une classe (cluster) entière.
   c) Un modèle de langage génératif.
   d) Une fonction de perte pour l'entraînement GPU.
   **[Réponse: b]** [Explication: Le "c" signifie *class-based*. Il permet d'extraire les mots-clés qui définissent l'identité d'un cluster. SOURCE: Livre p.149, Figure 5-13]

5. **Quel est l'avantage principal de la technique `KeyBERTInspired` par rapport au c-TF-IDF pur ?**
   a) Elle est beaucoup plus rapide.
   b) Elle utilise la similarité sémantique des embeddings pour s'assurer que les mots-clés sont vraiment proches du cœur du sujet.
   c) Elle permet de traduire les étiquettes automatiquement.
   d) Elle réduit la consommation de VRAM.
   **[Réponse: b]** [Explication: KeyBERTInspired vérifie que les mots les plus fréquents sont aussi sémantiquement cohérents avec le cluster. SOURCE: Livre p.158, Figure 5-22]

6. **Quelle technique permet de réduire la redondance dans les listes de mots-clés (éviter d'avoir "chat" et "chats" dans le même sujet) ?**
   a) UMAP
   b) HDBSCAN
   c) MMR (Maximal Marginal Relevance)
   d) Le Fine-tuning MLM
   **[Réponse: c]** [Explication: MMR cherche un équilibre entre la pertinence au sujet et la diversité entre les mots-clés choisis. SOURCE: Livre p.159]

7. **Combien de dimensions sont généralement recommandées pour UMAP avant de passer à l'étape de clustering avec HDBSCAN ?**
   a) 2 dimensions (pour l'affichage graphique)
   b) Entre 5 et 10 dimensions
   c) Exactement 768 dimensions
   d) 1 dimension
   **[Réponse: b]** [Explication: Réduire à 2 dimensions perd trop d'information pour le clustering. On réduit à 5-10 pour HDBSCAN, puis à 2 seulement pour le graphique. SOURCE: Livre p.142]

8. **Quel outil intégré à BERTopic permet de visualiser interactivement les documents et de voir leur contenu au survol de la souris ?**
   a) Matplotlib
   b) La méthode `visualize_documents()` basée sur Plotly
   c) Excel
   d) TensorBoard
   **[Réponse: b]** [Explication: Cette méthode permet d'explorer dynamiquement la "carte sémantique" générée. SOURCE: Livre p.155, Figure 5-18]

9. **Quelle est la différence fondamentale entre BERTopic et les méthodes classiques comme la LDA (Latent Dirichlet Allocation) ?**
   a) BERTopic est gratuit, pas la LDA.
   b) BERTopic utilise des embeddings contextuels (Transformers) au lieu de simples comptes de mots.
   c) LDA ne fonctionne que sur de très petits textes.
   d) BERTopic ne peut pas être utilisé sur GPU.
   **[Réponse: b]** [Explication: Grâce aux Transformers, BERTopic "comprend" le sens avant de compter, contrairement à la LDA qui est purement statistique. SOURCE: Livre p.148, p.151]

10. **Dans l'algorithme MMR, que permet de contrôler le paramètre `diversity` ?**
    a) Le nombre de documents par cluster.
    b) Le degré de différence forcée entre les mots-clés sélectionnés pour représenter un sujet.
    c) La vitesse de réduction UMAP.
    d) La langue du modèle.
    **[Réponse: b]** [Explication: Une diversité élevée force le modèle à choisir des mots-clés qui couvrent des angles différents du sujet. SOURCE: Livre p.160]

---

### 🔹 EXERCICE 1 : Pipeline de clustering complet (Niveau 1)

**Objectif** : Implémenter manuellement le pipeline (Embedding -> UMAP -> HDBSCAN) pour découvrir des structures dans un petit corpus.

```python
# --- CODE AVANT COMPLÉTION (QUESTION) ---
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

# --- RÉPONSE COMPLÈTE (CORRIGÉ) ---
# [SOURCE: Pipeline classique Livre p.139-142]

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

---

### 🔹 EXERCICE 2 : BERTopic avancé avec Reranking (Niveau 2)

**Objectif** : Utiliser la modularité de BERTopic pour affiner la représentation des sujets avec KeyBERTInspired.

```python
# --- CODE AVANT COMPLÉTION (QUESTION) ---
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired

# Supposons que nous avons une liste 'abstracts' de 1000 articles ArXiv
# abstracts = [...] 

# TÂCHE : Initialisez BERTopic et mettez à jour les étiquettes avec KeyBERTInspired.

# --- RÉPONSE COMPLÈTE (CORRIGÉ) ---
# [SOURCE: Représentation des sujets Livre p.157-158]

# 1. Création du modèle de base
topic_model = BERTopic(embedding_model="all-MiniLM-L6-v2", verbose=True)
topics, probs = topic_model.fit_transform(abstracts)

# 2. Sculpture sémantique avec KeyBERTInspired (Reranking)
# [SOURCE: Figure 5-20 p.157]
representation_model = KeyBERTInspired()

# 3. Mise à jour SANS recalculer les clusters (gain de temps !)
topic_model.update_topics(abstracts, representation_model=representation_model)

print("Nouveaux mots-clés (plus sémantiques) :")
print(topic_model.get_topic(0)[:5])
```

**Explications détaillées** :
*   **Résultats attendus** : Les mots-clés du sujet 0 doivent être plus précis (ex: "neural_networks" au lieu de "model").
*   **Justification** : `update_topics` permet de changer la "lentille" à travers laquelle on regarde les clusters existants. KeyBERTInspired recalcule la proximité entre les mots et le centre du cluster.

---

### 🔹 EXERCICE 3 : Visualisation et gestion du bruit (Niveau 3)

**Objectif** : Générer une visualisation et analyser le cluster des "outliers".

**Tâche** : 
1. Exécutez `topic_model.visualize_hierarchy()` sur vos résultats.
2. Identifiez le nombre de documents classés en `-1` (bruit) via `topic_model.get_topic_info()`.

**Réponse typique et analyse** :
*   **Action** : `topic_model.visualize_hierarchy()` affiche un dendrogramme montrant comment les sujets se regroupent.
*   **Interprétation du bruit (-1)** : ⚠️ **Avertissement du Professeur** : Si plus de 30% de vos documents sont en `-1`, cela signifie que vos paramètres `min_cluster_size` sont trop stricts ou que vos données sont trop disparates. 
*   **Justification** : BERTopic privilégie la "pureté" d'un sujet. Il préfère ne rien dire sur un document plutôt que de l'inclure de force dans un thème qui ne lui correspond pas. [SOURCE: Livre p.153]

---

**Mots-clés de la semaine** : Clustering, UMAP, HDBSCAN, BERTopic, c-TF-IDF, Outliers, MMR, KeyBERTInspired, Visualisation hiérarchique, Représentation sémantique.

**En prévision de la semaine suivante** : Nous allons apprendre à maîtriser l'interface entre l'humain et la machine : l'art du **Prompt Engineering**. Comment formuler vos demandes pour obtenir le meilleur des LLM ? [SOURCE: Detailed-plan.md]

**SOURCES COMPLÈTES** :
*   Livre : Alammar & Grootendorst (2024), *Hands-On LLMs*, Chapitre 5, p.137-164.
*   BERTopic Documentation : https://maartengr.github.io/BERTopic/
*   SBERT Clustering Guide : https://www.sbert.net/examples/applications/clustering/README.html
*   GitHub Officiel : chapter05 repository.

[/CONTENU SEMAINE 7]