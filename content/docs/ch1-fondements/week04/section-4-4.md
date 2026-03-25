---
title: "4.4 Tâches de classification avancées"
weight: 5
---

{{< katex />}}

## L'IA sans professeur : Le défi du monde réel
Bonjour à toutes et à tous ! Imaginez un instant que vous soyez parachuté dans une entreprise qui reçoit des milliers de retours clients chaque jour. Votre patron vous demande de les classer par urgence, mais il y a un problème de taille : vous n'avez aucune donnée déjà étiquetée. Pas un seul exemple "urgent" ou "normal" pour entraîner votre classifieur Scikit-Learn de la section précédente. Allez-vous passer vos nuits à étiqueter à la main ? 
> [!IMPORTANT]
🔑 **Je dois insister : avant de sortir vos étiqueteuses, sortez vos embeddings !**

Dans cette dernière section de la semaine, nous allons explorer la magie de la classification **Zero-shot** (zéro-exemple). C'est l'une des applications les plus puissantes des modèles de représentation. Nous allons apprendre à classifier du texte simplement en utilisant la géométrie de l'espace vectoriel. Respirez, nous allons voir comment transformer un nom de classe en une position GPS mathématique.

## L'intuition du Zero-shot : La comparaison sémantique
La classification classique (supervisée) demande au modèle d'apprendre la frontière entre deux nuages de points. La classification Zero-shot, elle, repose sur une idée brillante illustrée dans les **Figures 4-9 à 4-12**. 

{{< bookfig src="99.png" week="04" >}}
{{< bookfig src="100.png" week="04" >}}
<a id="fig-4-11"></a>
{{< bookfig src="101.png" week="04" >}}
{{< bookfig src="102.png" week="04" >}}

Le processus est le suivant :
1.  **Embedder le document** : Nous transformons la phrase du client en un vecteur avec BERT (ex: "Mon colis est arrivé cassé !").
2.  **Embedder les étiquettes** : C'est le coup de génie. Nous transformons les noms des classes eux-mêmes en vecteurs. Nous créons un vecteur pour le mot "Urgent" et un vecteur pour le mot "Normal".
3.  **Comparer les distances** : Le document appartient à la classe dont le vecteur est le plus "proche" du sien dans l'espace multidimensionnel. 

> [!IMPORTANT]
🔑 **La distinction non-négociable :** Dans ce scénario, nous ne classons pas par rapport à une règle apprise, mais par rapport à la proximité sémantique brute. C'est le modèle de langage qui "sait" déjà que "cassé" est sémantiquement plus proche d' "Urgent" que de "Normal".

## L'outil de mesure : La Similarité Cosinus
Comment la machine calcule-t-elle cette "proximité" ? Elle utilise généralement la **Similarité Cosinus** (nous allons la voir en détail dans la [**Semaine 06**]({{< relref "section-6-2.md" >}})), représentée en [**Figure 4-11**](#fig-4-11). 

> [!WARNING]
⚠️ **Attention : erreur fréquente ici !** On pourrait être tenté d'utiliser la distance euclidienne (la règle droite entre deux points). Mais en NLP, la longueur des vecteurs peut varier selon la richesse du texte. La similarité cosinus, elle, ne regarde que l'**angle** entre les vecteurs.

>*   Si l'angle est de 0°, le cosinus est de 1 : les textes sont sémantiquement identiques.
>*   Si l'angle est de 90°, le cosinus est de 0 : ils n'ont aucun rapport.
>*   Si l'angle est de 180°, le cosinus est de -1 : ils sont opposés.

**Analogie** : C'est comme deux boussoles. Peu importe si une aiguille est plus longue que l'autre ; si elles pointent toutes les deux vers le Nord, elles sont "similaires".

## Implémentation : Zero-shot avec Scikit-Learn
Voici comment mettre cela en œuvre très simplement. Nous allons utiliser `SentenceTransformer` pour les vecteurs et `cosine_similarity` de Scikit-Learn pour le calcul.

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. Chargement du modèle de représentation
model = SentenceTransformer("all-mpnet-base-v2")

# 2. Nos documents à classer (sans labels)
texts = [
    "I am absolutely furious about the delay!",
    "The weather is quite nice today for a walk.",
    "My computer won't turn on and I have a deadline."
]

# 3. Nos classes cibles (on les traite comme du texte !)
labels = ["technical support", "angry customer", "casual conversation"]

# 4. Encodage en vecteurs
text_embeddings = model.encode(texts)
label_embeddings = model.encode(labels)

# 5. Calcul de la matrice de similarité
# On compare chaque texte à chaque label
similarities = cosine_similarity(text_embeddings, label_embeddings)

# 6. Attribution de la classe la plus proche
for i, text in enumerate(texts):
    best_label_idx = np.argmax(similarities[i])
    print(f"Texte: '{text}' -> Classe: {labels[best_label_idx]}")

```

## L'art de nommer les classes : Le "Label Prompting"

> [!IMPORTANT]
🔑 **Je dois insister :** La performance du Zero-shot dépend énormément de la façon dont vous nommez vos classes. 

>Si vous nommez une classe "X12", BERT ne pourra rien faire. Si vous la nommez "Problème de facturation", il sera très efficace. 

> [!TIP]
Parfois, il est même préférable d'utiliser une petite **description** plutôt qu'un seul mot. Au lieu de "Sport", essayez "Un article traitant de compétitions sportives, d'athlètes ou de résultats de matchs". En enrichissant le vecteur de la classe, vous donnez plus de chances au document de s'y "aimanter".

## Limites et Bonnes Pratiques

> [!WARNING]
⚠️ **Ne tombez pas dans la paresse technologique!** Le Zero-shot a des limites claires :
1.  **Sensibilité au vocabulaire** : Si votre domaine utilise un jargon très spécifique que BERT n'a pas vu (ex: des codes d'erreur industriels), la similarité sera faible.
2.  **Performance** : Un modèle entraîné avec des labels (supervisé) sera TOUJOURS plus précis qu'un modèle Zero-shot. Le Zero-shot est une excellente solution de secours ou un point de départ pour pré-étiqueter vos données.
3.  **Coût de calcul** : Si vous avez 1000 classes, vous devez comparer chaque document à 1000 vecteurs de labels, ce qui peut ralentir le système.

## Éthique et Biais dans le choix des labels

> [!WARNING]
⚠️ **Éthique ancrée** : Mes chers étudiants, le choix des mots n'est jamais neutre.  

>En Zero-shot, c'est **vous** qui définissez l'espace sémantique. Si vous créez une classe "Sentiment agressif" au lieu de "Réclamation client", vous orientez déjà la façon dont l'IA va percevoir vos utilisateurs. 

> [!CAUTION]
🔑 **Conséquence éthique :** Les biais du modèle (vus en [**4.1**]({{< relref "section-4-1.md" >}})) vont interagir avec vos labels.

>Si le modèle associe statistiquement certains dialectes ou manières de parler à une classe négative, vous risquez de discriminer certains groupes sans même avoir entraîné le modèle. Auditez toujours les résultats du Zero-shot pour vérifier qu'une catégorie de population n'est pas systématiquement mal classée par pur préjugé statistique du modèle.

---
Nous avons bouclé la boucle ! De la structure rigide de BERT aux nuances fluides du Zero-shot, vous maîtrisez maintenant la science des modèles de représentation. Vous savez non seulement comment ils fonctionnent, mais comment les déployer quand les données manquent. La semaine prochaine, nous basculerons du côté de la création avec **les modèles de génération (GPT)**. Mais avant cela, rendez-vous au laboratoire pour mettre tout cela en pratique !

