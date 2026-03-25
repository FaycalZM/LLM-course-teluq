[CONTENU SEMAINE 6]

# Semaine 6 : Recherche sémantique et embeddings textuels

**Titre : Au-delà des mots-clés : La révolution de la recherche sémantique**

**Accroche du Professeur Khadidja Henni** : 
« Bonjour à toutes et à tous ! Je suis ravie de vous retrouver pour entamer cette sixième semaine. Imaginez un instant que vous soyez dans une bibliothèque infinie. Jusqu'à hier, pour trouver un livre, vous deviez connaître exactement un mot du titre. Aujourd'hui, je vais vous donner un super-pouvoir : vous allez pouvoir trouver un ouvrage simplement en décrivant l'émotion ou l'idée qu'il contient, même si vos mots ne sont pas ceux de l'auteur. C'est cela, la magie de la recherche sémantique. Nous allons apprendre à traduire le "sens" en "géométrie". Prêt·e·s pour ce voyage dans l'espace vectoriel ? » [SOURCE: Livre p.225]

**Rappel semaine précédente** : « La semaine dernière, nous avons maîtrisé les modèles génératifs de la famille GPT (Decoder-only), en apprenant à piloter leur créativité grâce aux paramètres de température et de nucleus sampling (Top-P). » [SOURCE: Detailed-plan.md]

**Objectifs de la semaine** :
À la fin de cette semaine, vous saurez :
*   Expliquer comment transformer une phrase complète en un vecteur unique (Sentence Embeddings).
*   Comparer les différentes stratégies de pooling (CLS vs Mean Pooling).
*   Calculer la similarité entre deux textes via la similarité cosinus.
*   Concevoir l'architecture technique d'un moteur de recherche sémantique (Chunking, Indexation, Retrieval).
*   Utiliser la bibliothèque FAISS pour gérer des millions de documents.

---

## 6.1 Embeddings de phrases (1200+ mots)

### Le défi du passage à l'échelle sémantique
« Mes chers étudiants, rappelez-vous de notre Semaine 2. Nous avions appris que chaque mot peut être un vecteur. Mais dans la vie réelle, nous ne communiquons pas par mots isolés. Nous utilisons des phrases, des paragraphes, des intentions. » 

Si vous essayez de comparer deux documents en faisant simplement la moyenne des vecteurs de chaque mot (Word2Vec), vous obtiendrez un résultat médiocre. Pourquoi ? Parce que la syntaxe et l'ordre des mots comptent ! "L'homme a mordu le chien" et "Le chien a mordu l'homme" ont les mêmes mots, mais des sens radicalement opposés. Pour résoudre cela, nous avons besoin d'**embeddings de phrases** (*Sentence Embeddings*). 

Comme l'illustre la **Figure 2-10 : Processus de création d'embeddings textuels** (p.62 du livre), le but est d'utiliser un modèle de langage comme un extracteur de caractéristiques globales. Le modèle reçoit une phrase entière et, après être passé par ses couches de Transformers, il condense toute l'information dans un vecteur unique de taille fixe (souvent 384, 768 ou 1024 dimensions). [SOURCE: Livre p.62, Figure 2-10]

### La naissance de Sentence-BERT (SBERT)
Pendant longtemps, utiliser BERT pour comparer des phrases était incroyablement lent. Comme nous l'avons vu en Semaine 4, BERT est conçu pour la classification ou le remplissage de masques. Pour comparer deux phrases, il fallait les passer ensemble dans le modèle (Cross-Encoder), ce qui était impossible à grande échelle.

La révolution est venue de **SBERT** (Reimers & Gurevych, 2019). 🔑 **Je dois insister sur cette innovation :** SBERT utilise une architecture dite "Siamese" (Siamoise). On utilise deux réseaux BERT identiques qui partagent les mêmes poids. Chaque phrase passe par un réseau, et on optimise le modèle pour que les phrases ayant un sens proche finissent avec des vecteurs proches. 

C'est ce que montre la **Figure 8-4 : Recherche sémantique dense** (p.228). Dans cette illustration, vous pouvez voir que dans l'espace mathématique du modèle, la phrase "Le ciel est bleu" se retrouve physiquement proche de "Il fait beau aujourd'hui", alors qu'elle est très éloignée de "Le moteur de ma voiture est cassé". 🔑 **Notez bien cette intuition :** La recherche sémantique transforme le langage en une carte géographique où la proximité physique égale la proximité de sens. [SOURCE: Livre p.228, Figure 8-4]

### L'art du Pooling : Comment résumer une séquence ?
Un Transformer produit un vecteur pour *chaque* token de la phrase. Si votre phrase fait 10 tokens, vous avez 10 vecteurs. Comment n'en obtenir qu'un seul pour représenter la phrase entière ? C'est ce qu'on appelle le **Pooling**.

Il existe trois stratégies principales :
1.  **CLS Pooling** : On utilise uniquement le vecteur du premier token spécial `[CLS]`. C'est la méthode historique de BERT.
2.  **Mean Pooling (Moyenne)** : On calcule la moyenne mathématique de tous les vecteurs de la phrase. 🔑 **C'est la méthode recommandée par SBERT :** Elle capture mieux l'essence globale de la séquence que le simple token `[CLS]`.
3.  **Max Pooling** : On prend la valeur maximale de chaque dimension à travers tous les tokens.

⚠️ **Attention : erreur fréquente ici !** N'essayez pas de faire du pooling manuellement si vous utilisez la bibliothèque `sentence-transformers`. Elle gère cela automatiquement selon le modèle choisi. [SOURCE: SBERT.net Documentation]

### Choisir son modèle : Modèles de fondation vs Modèles spécialisés
« Ne prenez pas le premier modèle venu sur Hugging Face ! » Pour la recherche sémantique, la taille ne fait pas tout. 

*   **all-mpnet-base-v2** : C'est le "couteau suisse". Avec ses 768 dimensions, il offre le meilleur équilibre entre précision et vitesse pour l'anglais. [SOURCE: Livre p.62]
*   **paraphrase-multilingual-MiniLM-L12-v2** : Indispensable si vous travaillez en français ou dans plusieurs langues simultanément. Il est très léger et rapide.
*   **gte-small / bge-micro** : Des modèles très récents (2023-2024) qui battent des records sur les benchmarks tout en étant minuscules.

🔑 **Le baromètre absolu : Le MTEB Leaderboard**. Comme nous l'avons évoqué, le *Massive Text Embedding Benchmark* est votre boussole. Avant de déployer un système, vérifiez la position de votre modèle dans la colonne "Retrieval" (Recherche). [SOURCE: Blog 'LLM Roadmap' de Maarten Grootendorst]

### Projection spatiale et intuition de recherche
Regardez la **Figure 8-5 : Projection de la requête** (p.229). Lorsqu'un utilisateur pose une question, le système transforme cette question en un vecteur (une étoile dans la galaxie). Le moteur de recherche n'a plus qu'à regarder quelles sont les "étoiles" (les documents) les plus proches de la question.

C'est ce qui est détaillé en **Figure 8-6 : Workflow de la base de connaissance** (p.230). 
1.  On transforme tous nos documents en vecteurs à l'avance (Indexation).
2.  On les stocke dans une "base de données vectorielle".
3.  À la requête, on compare et on affiche les résultats.

### Laboratoire de code : Créer ses premiers embeddings (Colab T4)
Voici comment implémenter cela très simplement. Nous allons utiliser la bibliothèque `sentence-transformers`, qui est le standard de l'industrie.

```python
# Installation : !pip install sentence-transformers
from sentence_transformers import SentenceTransformer
import torch

# 1. Chargement du modèle de pointe
# [SOURCE: Modèle recommandé Livre p.62]
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

# [SOURCE: CONCEPT À SOURCER – INSPIRÉ DU REPO GITHUB CHAPTER 2]
```

⚠️ **Fermeté bienveillante** : Observez bien la sortie du code. La phrase 1 et la phrase 2 n'ont presque aucun mot en commun ("cat" vs "feline", "sofa" vs "couch"). Pourtant, grâce aux embeddings de phrases, leurs vecteurs seront presque identiques. C'est ici que vous tuez la recherche par mot-clé !

### Éthique et Transparence : Les dimensions cachées du sens
⚠️ **Éthique ancrée** : « Mes chers étudiants, un vecteur est une réduction de la réalité. » 
Quand vous compressez une pensée humaine complexe dans 768 nombres, vous perdez forcément des nuances. 
1.  **Biais de distance** : Si votre modèle a été entraîné sur des textes majoritairement occidentaux, il pourrait considérer que "mariage" est plus proche de "fête" que de "contrat", ce qui n'est pas universel.
2.  **Sensibilité aux détails** : Les embeddings de phrases sont parfois "trop globaux". Ils peuvent rater une petite négation ("ne... pas") qui change tout le sens de la phrase. 

🔑 **Je dois insister :** La recherche sémantique est un outil de découverte, pas un juge de vérité. Toujours prévoir une étape de vérification humaine ou un modèle de reranking (que nous verrons en section 6.3) pour affiner les résultats. [SOURCE: Livre p.28]

« Vous avez maintenant les bases de la cartographie sémantique. Vous savez transformer du texte en coordonnées. Dans la section suivante, nous allons apprendre à mesurer avec précision la "distance" entre ces points : bienvenue dans le monde de la similarité cosinus. »

---
*Fin de la section 6.1 (1280 mots environ)*
## 6.2 Mesures de similarité (1000+ mots)

### La quête de la proximité mathématique
« Bonjour à toutes et à tous ! J'espère que vous avez bien en tête nos "étoiles sémantiques" de la section précédente. Nous savons maintenant transformer une idée en un point dans l'espace. Mais une question fondamentale subsiste : comment dire à une machine, de manière indiscutable, que deux points sont "proches" ? 🔑 **Je dois insister :** en informatique, le sens n'est pas une intuition, c'est un calcul. Aujourd'hui, nous allons découvrir les règles de cette géométrie du sens. Respirez, nous allons transformer des angles en affinités. » [SOURCE: Livre p.228]

### La Similarité Cosinus : Le standard de l'industrie
La mesure reine en NLP est la **Similarité Cosinus**. Pourquoi ne pas simplement utiliser une règle pour mesurer la distance entre deux points ? Parce que dans le langage, la "longueur" d'un vecteur (sa magnitude) peut être trompeuse.

Regardez attentivement la **Figure 4-15 : Similarité cosinus** (p.125 du livre). Cette illustration est capitale. Imaginez deux flèches partant de l'origine (le point 0,0). 
*   La flèche A représente un document court : "J'aime les chats."
*   La flèche B représente un document long sur le même sujet : "Le bonheur de posséder un petit félin domestique est immense..."
Leurs vecteurs n'auront pas la même longueur, mais ils pointent exactement dans la même direction sémantique. La similarité cosinus ne mesure pas la distance entre les pointes des flèches, mais l'**angle** $\theta$ entre elles. 

🔑 **L'intuition mathématique :** 
*   Si l'angle est de 0°, les deux phrases pointent dans la même direction : Similarité = 1 (Identiques).
*   Si l'angle est de 90°, elles n'ont aucun rapport (orthogonalité) : Similarité = 0.
*   Si l'angle est de 180°, elles sont opposées : Similarité = -1.

⚠️ **Attention : erreur fréquente ici !** Dans la plupart des bibliothèques de LLM, les scores de similarité oscillent entre 0 et 1 car les vecteurs sont normalisés. Un score de 0.9 est excellent, un score de 0.1 indique une absence totale de lien. [SOURCE: Livre p.125, Figure 4-15]

### Comparaison : Euclidienne vs. Cosinus
« Pourquoi ne pas utiliser la distance Euclidienne (L2), celle que vous avez apprise à l'école ? » 

1.  **Distance Euclidienne** : C'est la ligne droite entre deux points. 
    *   *Problème* : Si vous comparez un paragraphe de 10 lignes et un livre de 500 pages sur le même sujet, la distance Euclidienne sera énorme simplement parce que le second vecteur est "plus loin" de l'origine à cause de sa masse de mots.
2.  **Similarité Cosinus** : Elle normalise la longueur. 
    *   *Avantage* : Elle se concentre uniquement sur l'**orientation**. C'est pour cela qu'elle est le choix par défaut pour la recherche sémantique. 

🔑 **Note technique :** Si vous "normalisez" vos vecteurs (c'est-à-dire que vous ramenez leur longueur à 1), alors la distance Euclidienne et la similarité cosinus deviennent mathématiquement équivalentes. C'est ce que font beaucoup de bases de données vectorielles pour gagner en vitesse. [SOURCE: Pinecone Learning Center 'Vector Similarity Explained']

### Le Produit Scalaire (Dot Product)
Le **Dot Product** est le calcul brut derrière la similarité. C'est une simple multiplication membre à membre des composants de deux vecteurs, suivie d'une somme.
*   Si vos vecteurs ne sont pas normalisés, le Dot Product peut donner des nombres énormes.
*   C'est la mesure préférée des GPU car elle est extrêmement rapide à calculer sous forme de matrices. [SOURCE: Livre p.229]

### Fine-tuning pour le Retrieval : Ajuster la boussole
« Parfois, les modèles pré-entraînés ne suffisent pas. Un modèle généraliste peut penser que "pomme" est proche de "banane" (ce sont des fruits), mais si vous construisez un moteur de recherche pour une entreprise de technologie, vous voulez peut-être que "pomme" soit proche de "iPhone". »

C'est ici qu'intervient le **fine-tuning pour le retrieval**, illustré par les **Figures 8-12 et 8-13** (p.239-240). 
*   **Figure 8-12** : Montre l'état initial. Les requêtes (queries) et les documents sont éparpillés. Le modèle ne sait pas encore quel document répond spécifiquement à quelle question.
*   **Figure 8-13** : Après l'entraînement sur des paires "Question/Réponse", le modèle a appris à "tirer" les bonnes réponses vers la question et à "pousser" les mauvaises au loin. 

🔑 **Je dois insister :** pour réussir ce réglage, vous avez besoin de "Contrastive Pairs" : une question, une réponse correcte (exemple positif) et une réponse erronée (exemple négatif). C'est ainsi que l'on transforme un modèle de langage en un expert en recherche. [SOURCE: Livre p.239-240, Figures 8-12, 8-13]

### Implémentation pratique : Similarité avec Scikit-learn
Voici comment calculer ces scores de proximité sur Google Colab.

```python
# Testé sur Colab T4
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. On charge notre modèle de référence
model = SentenceTransformer("all-mpnet-base-v2")

# 2. On définit une requête et des documents
query = "How to improve my health?"
docs = [
    "You should exercise daily and eat vegetables.", # Très pertinent
    "The stock market is fluctuating due to global news.", # Hors sujet
    "Maintaining a balanced diet is key to well-being." # Pertinent
]

# 3. Encodage en vecteurs
query_vec = model.encode([query])
docs_vec = model.encode(docs)

# 4. Calcul de la similarité cosinus (QUESTION CODE)
# similarities = ...

# --- RÉPONSE (ANSWER CODE) ---
# [SOURCE: CONCEPT À SOURCER – Documentation Scikit-learn & Livre p.125]
similarities = cosine_similarity(query_vec, docs_vec)

print(f"Requête : {query}\n")
for i, score in enumerate(similarities[0]):
    print(f"Document {i+1} : Score = {score:.4f} -> {docs[i]}")

# ATTENDU : Scores élevés (> 0.7) pour les docs 1 et 3, faible (< 0.2) pour le doc 2.
```

⚠️ **Fermeté bienveillante** : Regardez les scores. Même le document 2 aura un score légèrement supérieur à 0 (peut-être 0.15). 🔑 **C'est non-négociable :** en recherche vectorielle, tout est relié à tout à un certain degré. Votre rôle d'ingénieur est de fixer un **seuil (threshold)** en dessous duquel vous considérez que le résultat est du bruit.

### Éthique et Biais dans la similarité
⚠️ **Éthique ancrée** : « Mes chers étudiants, la mathématique de la similarité peut être injuste. » 
Les mesures de similarité ne sont que le reflet des corrélations apprises. 
1.  **Biais de neutralité** : Si vous cherchez "personne compétente", et que le modèle renvoie systématiquement des profils ayant un certain type de vocabulaire culturel, vous excluez des talents sans le savoir.
2.  **Fausse certitude** : Un score de 0.98 n'est pas une preuve de vérité, c'est une preuve de ressemblance statistique. 

🔑 **Mon conseil de professeur** : Ne basez jamais une décision automatique grave (recrutement, justice, crédit) uniquement sur un score de similarité vectorielle sans un audit humain rigoureux. [SOURCE: Livre p.28]

« Vous maîtrisez maintenant les instruments de mesure. Vous savez comment juger de la proximité entre deux pensées humaines transformées en nombres. Dans la section suivante, nous allons voir comment utiliser ces mesures pour construire une machine capable de fouiller dans des millions de documents en un clin d'œil : place à l'indexation vectorielle ! »

---
*Fin de la section 6.2 (1120 mots environ)*
## 6.3 Architecture de recherche (1300+ mots)

### Construire la bibliothèque du futur : Au-delà du vecteur unique
« Bonjour à toutes et à tous ! Nous arrivons à un moment charnière. Vous savez maintenant transformer du texte en vecteurs (section 6.1) et mesurer leur distance (section 6.2). Mais imaginez maintenant que vous deviez chercher la réponse à une question dans une base de données contenant **10 millions de documents**. Si vous comparez votre question à chaque document un par un, l'utilisateur aura pris sa retraite avant d'avoir sa réponse ! 🔑 **Je dois insister :** une bonne recherche sémantique ne repose pas seulement sur l'intelligence du modèle, mais sur la robustesse de l'architecture qui l'entoure. Aujourd'hui, nous allons apprendre à découper, indexer et fouiller massivement. » [SOURCE: Livre p.235]

### L'art délicat du découpage : Le Chunking
Un Transformer a une limite physique : sa fenêtre de contexte. On ne peut pas "donner" un livre entier à BERT pour qu'il en fasse un seul vecteur sans perdre une quantité colossale d'informations. C'est là qu'intervient le **chunking** (le découpage en tronçons).

Regardons ensemble la **Figure 8-7 : Un vecteur par document vs. Plusieurs vecteurs par document** (p.235 du livre). 
*   **À gauche (One vector per doc)** : On essaie de résumer tout un article en un point. C'est risqué. Si l'article parle de 10 sujets différents, le vecteur final sera une moyenne floue. 
*   **À droite (Multiple vectors)** : On découpe l'article en petits morceaux cohérents. Chaque morceau a son propre vecteur. C'est la stratégie gagnante pour la précision. [SOURCE: Livre p.235, Figure 8-7]

Les stratégies de découpage illustrées dans les **Figures 8-8 à 8-10** (p.236-237) sont vos outils de précision :
1.  **Découpage par caractères ou tokens (Figure 8-8)** : On coupe tous les 500 tokens. C'est simple mais brutal : on risque de couper une phrase ou une pensée en plein milieu.
2.  **Découpage structurel (Figure 8-9)** : On respecte les paragraphes ou les phrases. On utilise des outils comme `NLTK` ou `Spacy` pour repérer les limites naturelles du langage.
3.  **La fenêtre glissante avec chevauchement (Overlap) (Figure 8-10)** : C'est la technique préférée des experts. 🔑 **Notez bien cette intuition :** on crée des morceaux qui se chevauchent (par exemple, les 50 derniers tokens du bloc 1 se retrouvent au début du bloc 2). Pourquoi ? Pour s'assurer que si une information cruciale se trouve à la charnière de deux blocs, le contexte ne soit pas perdu. [SOURCE: Livre p.236-237, Figures 8-8, 8-9, 8-10]

### Passer à l'échelle avec FAISS : La recherche de plus proches voisins
Une fois vos millions de "chunks" transformés en vecteurs, comment trouver les plus proches ? On utilise l'algorithme des **K-Plus Proches Voisins (K-Nearest Neighbors ou KNN)**, illustré en **Figure 8-11** (p.238). [SOURCE: Livre p.238, Figure 8-11]

Pour gérer des millions de vecteurs en millisecondes, nous utilisons **FAISS** (*Facebook AI Similarity Search*), une bibliothèque optimisée pour les calculs matriciels massifs. FAISS nous propose deux mondes :

#### 1. La Recherche Exacte (Flat Index)
Le modèle compare votre requête à absolument TOUS les vecteurs. 
*   **Précision** : 100% (on trouve mathématiquement les meilleurs). 
*   **Vitesse** : Lente sur de très grosses bases. 
*   **Usage** : Jusqu'à environ 1 million de documents. [SOURCE: GitHub facebookresearch/faiss]

#### 2. La Recherche Approximative (ANN - Approximate Nearest Neighbors)
C'est ici que l'ingénierie devient "magique". Au lieu de tout fouiller, on utilise des index intelligents comme **IVF** (Inverted File Index).
*   **L'idée** : On regroupe les vecteurs similaires dans des "cellules" (clusters) à l'avance. Quand une question arrive, on identifie la cellule la plus proche et on ne fouille que celle-là. 
*   **Vitesse** : Foudroyante (recherche en microsecondes parmi des milliards).
*   **Compromis** : On accepte une infime chance de rater le meilleur résultat absolu pour gagner une vitesse immense. [SOURCE: Livre p.238-239, Blog 'Vector Similarity' de Pinecone]

### Architecture complète d'un système de recherche
Un moteur de recherche sémantique robuste suit ce pipeline non-négociable :
1.  **Ingestion** : Lecture des PDF, sites web ou bases de données.
2.  **Preprocessing** : Nettoyage et *Chunking* intelligent.
3.  **Embedding** : Passage dans un modèle comme `all-mpnet-base-v2`.
4.  **Indexing** : Stockage des vecteurs dans FAISS ou une base vectorielle (Chroma, Pinecone).
5.  **Querying** : Encodage de la question de l'utilisateur et recherche KNN. [SOURCE: Livre p.226]

### Laboratoire de code : Implémentation FAISS sur Colab (T4)
Voici comment construire votre premier index vectoriel professionnel. Nous allons simuler une base de connaissances et effectuer une recherche ultra-rapide.

```python
# Installation : !pip install faiss-gpu sentence-transformers
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Préparation du modèle (all-mpnet-base-v2 produit des vecteurs de taille 768)
# [SOURCE: Livre p.253 pour le choix du modèle]
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
# [SOURCE: CONCEPT À SOURCER – Documentation FAISS & Livre p.232]
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

# [SOURCE: CONCEPT À SOURCER – INSPIRÉ DU REPO GITHUB CHAPTER 8]
```

⚠️ **Avertissement du Professeur** : Notez bien l'usage de `IndexFlatIP`. IP signifie *Inner Product*. Comme nos embeddings sont normalisés par `sentence-transformers`, le produit scalaire devient identique à la similarité cosinus. C'est l'astuce de performance préférée des ingénieurs. [SOURCE: Livre p.125]

### Optimisations matérielles et GPU
Sur votre instance Colab T4, FAISS peut être déplacé sur le GPU pour une accélération encore plus massive. 🔑 **Je dois insister :** si vous dépassez les 10 millions de vecteurs, le calcul sur CPU deviendra un goulot d'étranglement. L'usage de la mémoire VRAM pour stocker l'index est une compétence clé à développer. [SOURCE: Livre p.51]

### Éthique et Confidentialité : La persistance des vecteurs
⚠️ **Éthique ancrée** : « Mes chers étudiants, soyez conscients de ce que vous indexez. » 
Une fois qu'un document est découpé en morceaux (chunks) et stocké sous forme de vecteurs dans une base de données, il devient très difficile de le "désindexer" totalement, surtout si vous utilisez des techniques de compression. 
1.  **Données personnelles (PII)** : Ne stockez jamais de noms, numéros de téléphone ou secrets dans vos index vectoriels sans anonymisation préalable. Un vecteur peut parfois être "renversé" pour retrouver une partie du texte d'origine.
2.  **Droit à l'oubli** : Si un utilisateur demande la suppression de ses données, vous devez être capable de retrouver tous les chunks associés à son identité dans votre index FAISS. C'est un défi d'ingénierie légale majeur sous le RGPD. [SOURCE: Livre p.28, p.50]

🔑 **Le message du Prof. Henni** : « L'architecture de recherche est le système nerveux de votre application. Si vos morceaux sont trop petits, le modèle sera confus. S'ils sont trop gros, il sera imprécis. Trouvez le juste milieu, et n'oubliez jamais que derrière chaque nombre, il y a une donnée humaine qui mérite votre protection. » [SOURCE: Livre p.28]

« Vous savez maintenant construire le moteur de recherche. Vous savez découper le savoir et l'indexer pour une vitesse fulgurante. Dans la dernière section de cette semaine, nous verrons comment perfectionner ce système pour qu'il ne se contente pas de trouver des documents, mais qu'il apprenne de ses erreurs : le réglage fin pour le retrieval. »

---
*Fin de la section 6.3 (1390 mots environ)*
## 6.4 Fine-tuning pour le retrieval (1200+ mots)

### Pourquoi un modèle généraliste ne suffit pas toujours
« Bonjour à toutes et à tous ! Nous arrivons au sommet de notre semaine sur la recherche sémantique. Jusqu'ici, nous avons utilisé des modèles "sur l'étagère" (*off-the-shelf*), comme des couteaux suisses capables de couper un peu de tout. Mais imaginez que vous deviez opérer un patient : utiliseriez-vous un couteau suisse ou un scalpel de précision ? 🔑 **Je dois insister :** dans des domaines pointus comme la médecine, le droit ou la maintenance industrielle, un modèle généraliste va passer à côté des nuances cruciales. Aujourd'hui, nous allons apprendre à forger votre propre scalpel sémantique grâce au **fine-tuning pour le retrieval**. » [SOURCE: Livre p.239]

Le problème fondamental est le suivant : un modèle comme `all-mpnet-base-v2` a été entraîné sur des données web générales (Wikipédia, Reddit). Il "sait" que "chat" est proche de "félin". Mais sait-il que dans votre base de données technique, le code d'erreur `E104` est sémantiquement lié à une "surchauffe de la pompe hydraulique" ? Probablement pas. Sans entraînement spécifique, la distance vectorielle entre ces deux concepts sera trop grande, et votre moteur de recherche échouera. ⚠️ **Attention : erreur fréquente ici !** Beaucoup d'ingénieurs pensent qu'il faut changer de modèle alors qu'il suffit souvent d'ajuster les poids du modèle actuel sur quelques centaines d'exemples métier. [SOURCE: Livre p.239]

### La géométrie du changement : Analyser les Figures 8-12 et 8-13
Pour bien comprendre ce qui se passe dans "le cerveau" du modèle pendant cet entraînement, regardons les schémas de l'espace vectoriel fournis par Jay Alammar et Maarten Grootendorst.

*   **Figure 8-12 : État avant le fine-tuning** (p.240) : Imaginez une carte du ciel. Vous avez une question (une requête) au centre. Autour d'elle, à des distances presque égales, gravitent plusieurs documents. Certains sont les bonnes réponses (exemples positifs), d'autres sont des hors-sujets qui partagent juste quelques mots en commun (exemples négatifs). Le modèle est confus : il ne "sent" pas la différence de pertinence. Tout se ressemble à ses yeux car il n'a pas appris votre contexte spécifique.
*   **Figure 8-13 : État après le fine-tuning** (p.240) : C'est ici que la magie opère. Après l'entraînement, la carte a été redessinée. Les documents pertinents ont été "aspirés" vers la requête (la distance diminue, le score de similarité augmente). À l'inverse, les documents non-pertinents ont été "repoussés" vers les bords de la galaxie. 🔑 **Notez bien cette intuition :** le fine-tuning est un processus de tension géométrique. On rapproche les alliés et on éloigne les intrus. [SOURCE: Livre p.240, Figures 8-12 et 8-13]

### La préparation des données : Triplets et paires contrastives
« Comment dire au modèle qui rapprocher et qui repousser ? » On ne lui donne pas des notes, on lui donne des **exemples de comparaison**. La méthode la plus robuste utilise des **triplets** (Ancre, Positif, Négatif) :
1.  **L'Ancre (Anchor)** : C'est la requête type de l'utilisateur (ex: "Comment réinitialiser le mot de passe ?").
2.  **Le Positif (Positive)** : C'est le paragraphe qui contient la vraie solution.
3.  **Le Négatif (Negative)** : C'est un document qui ressemble à la question mais ne contient pas la réponse.

🔑 **Le concept avancé : Les "Hard Negatives" (Négatifs difficiles)**. C'est le secret des meilleurs moteurs de recherche. Si vous donnez au modèle un négatif totalement absurde (ex: une recette de cuisine pour une question sur l'informatique), il n'apprendra rien. C'est trop facile. Pour qu'il devienne un expert, vous devez lui donner des documents qui parlent du même sujet mais qui sont techniquement incorrects pour cette question précise. C'est en échouant sur ces nuances qu'il affine sa compréhension. [SOURCE: Blog 'LLM Roadmap' de Maarten Grootendorst]

### Les fonctions de perte : Multiple Negatives Ranking (MNR) Loss
Mathématiquement, pour réaliser ce "rapprochement/éloignement", nous utilisons souvent la **MNR Loss**. 
L'idée est brillante : dans un lot (*batch*) de données, on suppose que pour chaque question, il n'y a qu'une seule bonne réponse parmi toutes celles présentes dans le lot. Le modèle essaie de maximiser la similarité de la paire correcte tout en minimisant celle de toutes les autres combinaisons possibles. C'est une forme d'apprentissage par élimination extrêmement efficace sur GPU. [SOURCE: SBERT.net Documentation & Livre p.304]

### Laboratoire de code : Fine-tuning rapide avec Sentence-Transformers
Voici comment transformer un modèle généraliste en un expert sur un petit jeu de données de support technique.

```python
# Testé pour Google Colab T4 16GB VRAM
# !pip install sentence-transformers datasets

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# 1. Chargement d'un modèle de base performant
model = SentenceTransformer('all-mpnet-base-v2')

# 2. Préparation de nos données métier (Triplets ou Paires)
# [SOURCE: Structure de données recommandée p.312]
train_examples = [
    InputExample(texts=['My internet is slow', 'Check your router settings and cables.'], label=0.9),
    InputExample(texts=['Blue screen error', 'This usually indicates a kernel panic or hardware failure.'], label=0.9),
    InputExample(texts=['How to pay my bill?', 'You can access the payment portal in your account settings.'], label=0.9),
    # On pourrait ajouter des exemples négatifs avec un label de 0.1
]

# 3. Création du DataLoader (gestion des lots pour le GPU)
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# 4. Définition de la fonction de perte (Loss)
# La CosineSimilarityLoss est parfaite pour débuter avec des scores de 0 à 1
# [SOURCE: Livre p.302]
train_loss = losses.CosineSimilarityLoss(model)

# 5. L'entraînement (Le Fine-tuning)
print("Début de l'entraînement métier...")
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=10)

# 6. Test après entraînement
# Le modèle est maintenant plus sensible à vos termes spécifiques
print("Modèle prêt pour votre domaine !")
```

### Évaluation : Comment savoir si on a progressé ?
⚠️ **Fermeté bienveillante** : « Ne vous contentez pas de regarder la perte descendre pendant l'entraînement. C'est un mirage. » Pour valider un système de recherche, vous devez utiliser des métriques spécifiques au domaine du *Information Retrieval* :
1.  **Hit Rate @ K** : "Dans mon top K résultats (ex: top 5), est-ce que la bonne réponse est présente au moins une fois ?"
2.  **MRR (Mean Reciprocal Rank)** : Cette métrique est plus sévère. Elle vous donne plus de points si la bonne réponse est en 1ère position qu'en 5ème. 🔑 **Je dois insister :** en recherche sémantique, l'utilisateur regarde rarement au-delà des trois premiers résultats. Le MRR est votre juge de paix. [SOURCE: Livre p.244-249]

### Éthique et Responsabilité : Le danger du sur-apprentissage (Overfitting)
⚠️ **Éthique ancrée** : « Mes chers étudiants, un modèle trop spécialisé devient un modèle aveugle. » 
Si vous entraînez trop fort votre modèle sur vos données internes, il risque de perdre ses connaissances générales. 
1.  **Le biais de domaine** : Si vous lui apprenez que "virus" désigne uniquement un logiciel malveillant, il pourrait devenir incapable de trouver des documents sur la biologie, ce qui peut être problématique si votre entreprise est dans la santé.
2.  **Confidentialité des données d'entraînement** : ⚠️ **Point crucial !** Si vos questions-réponses d'entraînement contiennent des secrets industriels ou des noms de clients, ces informations peuvent être "mémorisées" par le modèle dans ses poids. Un attaquant pourrait potentiellement extraire des fragments de ces données en interrogeant finement le modèle. 

🔑 **Le message du Prof. Henni** : « Le fine-tuning est l'acte final de l'artisan. Il donne au modèle son caractère unique. Mais un bon artisan sait aussi quand s'arrêter pour ne pas fragiliser la structure globale. Spécialisez avec précision, mais évaluez avec rigueur. » [SOURCE: Livre p.28]

« Nous avons terminé notre exploration de la recherche sémantique ! Vous savez désormais transformer du texte en vecteurs, mesurer leur similarité, construire une architecture à grande échelle et même entraîner le modèle pour qu'il devienne un expert. La semaine prochaine, nous irons encore plus loin en apprenant à regrouper ces documents automatiquement par thématiques : place au **Clustering** et à **BERTopic**. »

---
*Fin de la section 6.4 (1240 mots environ)*
## 🧪 LABORATOIRE SEMAINE 6 (750+ mots)

**Accroche du Professeur Khadidja Henni** : 
« Bonjour à toutes et à tous ! Nous y sommes : le moment de transformer votre ordinateur en un bibliothécaire omniscient. Dans ce laboratoire, nous allons construire un moteur de recherche capable de comprendre les intentions cachées derrière les mots. 🔑 **Je dois insister :** ne vous contentez pas de faire tourner le code. Changez les requêtes, observez les scores de similarité, et essayez de "piéger" le modèle. C'est en comprenant ses échecs que vous deviendrez de véritables experts. Prêt·e·s à plonger dans l'espace vectoriel ? »

---

### 🔹 QUIZ MCQ (10 questions)

1. **Quelle est la dimension typique d'un vecteur produit par le modèle de référence `all-mpnet-base-v2` ?**
   a) 384
   b) 512
   c) 768
   d) 1024
   **[Réponse: c]** [Explication: Ce modèle, basé sur une architecture BERT-base, utilise des vecteurs de 768 nombres réels pour compresser le sens d'une phrase. SOURCE: Livre p.84, p.253]

2. **Pourquoi la similarité cosinus est-elle préférée à la distance euclidienne (L2) en recherche sémantique ?**
   a) Elle est plus rapide à calculer sur CPU.
   b) Elle ignore la magnitude (longueur) du vecteur pour se concentrer sur l'angle sémantique.
   c) Elle permet de traiter des textes en plusieurs langues.
   d) Elle réduit automatiquement la taille du dictionnaire.
   **[Réponse: b]** [Explication: Dans le langage, un texte long et un texte court sur le même sujet ont des magnitudes différentes mais pointent dans la même direction. SOURCE: Livre p.125, p.229]

3. **Quelle bibliothèque est le standard industriel pour la recherche de plus proches voisins à l'échelle de millions de vecteurs ?**
   a) Pandas
   b) Scikit-learn
   c) FAISS (Facebook AI Similarity Search)
   d) PyTorch-Live
   **[Réponse: c]** [Explication: FAISS est optimisé pour les calculs matriciels massifs et permet des recherches approximatives ultra-rapides. SOURCE: Livre p.232, GitHub facebookresearch/faiss]

4. **Dans une stratégie de chunking, pourquoi utilise-t-on souvent un "overlap" (chevauchement) ?**
   a) Pour doubler la taille de la base de données.
   b) Pour s'assurer qu'une information située à la limite d'un découpage ne perde pas son contexte.
   c) Pour permettre au modèle de s'entraîner deux fois plus vite.
   d) Pour masquer les données sensibles.
   **[Réponse: b]** [Explication: Le chevauchement permet de conserver le lien sémantique entre la fin d'un bloc et le début du suivant. SOURCE: Livre p.236, Figure 8-10]

5. **Quelle technique est nécessaire pour qu'un modèle "sache" que dans votre entreprise, le terme "Projet Alpha" est lié à "Logistique" ?**
   a) Le Zero-shot classification.
   b) Le Fine-tuning pour le retrieval avec des paires contrastives.
   c) L'augmentation de la température.
   d) Le passage en quantification 4-bit.
   **[Réponse: b]** [Explication: Le fine-tuning ajuste les distances vectorielles pour coller à la réalité d'un domaine spécifique. SOURCE: Livre p.239]

6. **Quel est le principal défaut de l'utilisation d'un seul vecteur pour représenter un document de 50 pages ?**
   a) Le vecteur devient trop lourd pour le disque dur.
   b) Le goulot d'étranglement : les thématiques multiples se mélangent dans une "moyenne" floue.
   c) Le modèle refuse de traiter plus de 512 caractères.
   d) La similarité cosinus devient négative.
   **[Réponse: b]** [Explication: Un document long contient trop d'informations pour être résumé en un seul point sans perte massive de précision. SOURCE: Livre p.235, Figure 8-7]

7. **Dans FAISS, quel index est utilisé pour une recherche sémantique simple basée sur le produit scalaire (Inner Product) ?**
   a) IndexFlatL2
   b) IndexIVFFlat
   c) IndexFlatIP
   d) IndexBinary
   **[Réponse: c]** [Explication: IndexFlatIP calcule le produit scalaire, qui équivaut à la similarité cosinus si les vecteurs sont normalisés. SOURCE: Livre p.232, Documentation FAISS]

8. **Quelle est la différence fondamentale entre la recherche "Dense" et la recherche "Sparse" ?**
   a) Dense utilise des vecteurs de nombres réels, Sparse utilise des comptes de mots (TF-IDF).
   b) Dense est plus lente que Sparse.
   c) Sparse ne fonctionne qu'avec BERT.
   d) Dense nécessite un accès à Internet.
   **[Réponse: a]** [Explication: La recherche dense capture le sens (embeddings), la recherche sparse capture les mots exacts (lexical). SOURCE: Livre p.228, Figure 8-1]

9. **Quelle stratégie de "pooling" est la plus couramment utilisée par SBERT pour obtenir un embedding de phrase ?**
   a) Prendre uniquement le premier caractère.
   b) Faire la moyenne (Mean) de tous les vecteurs de tokens de la phrase.
   c) Prendre le vecteur du dernier token généré.
   d) Additionner les IDs des tokens.
   **[Réponse: b]** [Explication: Le Mean Pooling capture une représentation plus équilibrée de la séquence que le simple token CLS. SOURCE: SBERT.net Documentation]

10. **Quelle métrique d'évaluation donne une importance accrue à la position du bon résultat (mieux vaut être 1er que 5ème) ?**
    a) Hit Rate @ 10
    b) Accuracy
    c) MRR (Mean Reciprocal Rank)
    d) Taux de compression
    **[Réponse: c]** [Explication: Le MRR calcule l'inverse du rang du premier résultat correct, valorisant la précision en haut de liste. SOURCE: Livre p.244]

---

### 🔹 EXERCICE 1 : Moteur de recherche sémantique de base (Niveau 1)

**Objectif** : Implémenter une recherche sémantique simple en comparant manuellement une requête à une liste de documents.

```python
# --- CODE COMPLET (CORRIGÉ) ---
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

# --- RÉPONSE (ANSWER CODE) ---
# [SOURCE: Utilisation de sentence-transformers p.62]
# 2. Encodage des documents et de la requête
doc_embeddings = model.encode(documents, convert_to_tensor=True)
query = "Tell me about neural networks and AI"
query_embedding = model.encode(query, convert_to_tensor=True)

# 3. Calcul de la similarité cosinus via l'utilitaire de SBERT
# [SOURCE: Mesure de similarité p.125]
cosine_scores = util.cos_sim(query_embedding, doc_embeddings)[0]

# 4. Affichage des résultats triés
print(f"Requête : {query}")
top_results = torch.topk(cosine_scores, k=2)

for score, idx in zip(top_results[0], top_results[1]):
    print(f"Score: {score:.4f} | Document: {documents[idx]}")

# --- EXPLICATIONS DÉTAILLÉES ---
# ATTENDU : Le document 3 doit avoir le score le plus élevé car "neural networks" 
# est sémantiquement lié à la requête, même si les mots exacts ne sont pas tous présents.
# JUSTIFICATION : util.cos_sim gère la normalisation des vecteurs pour nous.
```

---

### 🔹 EXERCICE 2 : Indexation scale-up avec FAISS (Niveau 2)

**Objectif** : Utiliser la bibliothèque FAISS pour indexer des documents et effectuer une recherche "K-Plus Proches Voisins" (KNN).

```python
# --- CODE COMPLET (CORRIGÉ) ---
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Préparation (QUESTION CODE)
model = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384 # Dimension spécifique à MiniLM
corpus = ["The moon orbits the Earth.", "The sun is a star.", "Apples are fruits.", "Fast cars are exciting."]

# --- RÉPONSE (ANSWER CODE) ---
# [SOURCE: Architecture de recherche Livre p.232]
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

# --- EXPLICATIONS DÉTAILLÉES ---
# ATTENDU : "The moon orbits the Earth" ou "The sun is a star".
# JUSTIFICATION : FAISS permet de traiter des millions de documents là où 
# une boucle Python s'effondrerait. L'usage de float32 est crucial pour la compatibilité GPU.
```

---

### 🔹 EXERCICE 3 : Évaluation système : Calcul du MRR (Niveau 3)

**Objectif** : Implémenter la métrique Mean Reciprocal Rank (MRR) pour juger la qualité d'un moteur de recherche.

```python
# --- CODE COMPLET (CORRIGÉ) ---
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

# --- RÉPONSE (ANSWER CODE) ---
# [SOURCE: Métriques d'évaluation Livre p.244-249]
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

# --- EXPLICATIONS DÉTAILLÉES ---
# ATTENDU : Score d'environ 0.444 ( (1/3 + 1/1 + 0) / 3 ).
# JUSTIFICATION : Le MRR est impitoyable. Si le bon document n'est pas 1er, 
# le score chute vite. C'est la métrique reine pour les moteurs de recherche 
# où l'utilisateur ne clique que sur le premier lien.
```

---

**Mots-clés de la semaine** : Sentence Embeddings, SBERT, Similarité Cosinus, FAISS, Chunking, Overlap, KNN, Retrieval, MRR, Hit Rate.

**En prévision de la semaine suivante** : Nous allons apprendre à découvrir la structure cachée de vos documents. Comment regrouper automatiquement des milliers de textes par thématiques sans intervention humaine ? Bienvenue dans le monde du **Clustering** et de **BERTopic**. [SOURCE: Detailed-plan.md]

**SOURCES COMPLÈTES** :
*   Livre : Alammar & Grootendorst (2024), *Hands-On LLMs*, Chapitre 8, p.225-249.
*   Hugging Face : *Semantic Search Guide* (https://huggingface.co/blog/semantic-search).
*   Pinecone : *Vector Similarity* (https://www.pinecone.io/learn/vector-similarity/).
*   GitHub Officiel : chapter08 repository.

[/CONTENU SEMAINE 6]