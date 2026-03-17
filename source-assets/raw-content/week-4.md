---
# WEEK: 4
# TITLE: Semaine 4 : Modèles de représentation (encoder-only)
# CHAPTER_FIGURES: [25, 26, 27, 89, 90, 91, 94, 95, 99, 100, 101, 102]
# COLAB_NOTEBOOKS: []
---
[CONTENU SEMAINE 4]
# Semaine 4 : Modèles de représentation (encoder-only)

**Titre : BERT et ses dérivés : L'ère des modèles de représentation**

**Accroche du Professeur Khadidja Henni** : 
« Bonjour à toutes et à tous ! Je suis ravie de vous retrouver pour entamer notre deuxième grand chapitre : la Science des LLM. Jusqu'ici, nous avons beaucoup parlé de génération, de ces modèles qui "parlent". Mais aujourd'hui, nous allons nous intéresser aux modèles qui "écoutent" et "comprennent" avec une précision chirurgicale. Bienvenue dans le monde de **BERT** et des architectures *Encoder-only*. Respirez, nous allons voir comment ces modèles parviennent à capturer l'essence même d'une phrase. » [SOURCE: Livre p.111]

**Rappel semaine précédente** : « La semaine dernière, nous avons plongé dans les entrailles de l'architecture Transformer, en décortiquant le mécanisme d'attention, le positionnement RoPE et le fonctionnement du forward pass. » [SOURCE: Detailed-plan.md]

**Objectifs de la semaine** :
À la fin de cette semaine, vous saurez :
*   Expliquer la structure interne de l'architecture *Encoder-only*.
*   Comprendre le rôle crucial du token `[CLS]` pour la classification.
*   Détailler le processus d'apprentissage par *Masked Language Modeling* (MLM).
*   Comparer les variantes de la famille BERT (RoBERTa, DistilBERT, DeBERTa).
*   Implémenter une pipeline de classification de sentiments.

---

## 4.1 La famille BERT (1100+ mots)

### L'intuition de l'encodeur : Lire entre les lignes
« Imaginez que vous lisiez une phrase. Pour en comprendre le sens, vous ne vous contentez pas de regarder les mots qui précèdent. Votre regard fait des va-et-vient, n'est-ce pas ? » C'est exactement ce que fait BERT (*Bidirectional Encoder Representations from Transformers*). Contrairement à la famille GPT que nous avons croisée et qui est "aveugle" aux mots futurs, BERT est **bidirectionnel**.

Comme vous pouvez l'observer sur la **Figure 1-21 : Architecture BERT base** (p.18 du livre), un modèle BERT-base est une pile de 12 blocs "encodeurs". 🔑 **Je dois insister sur cette distinction :** ici, il n'y a pas de mécanisme de masquage de l'attention comme dans les décodeurs. Chaque mot (token) peut "voir" tous les autres mots de la phrase, qu'ils soient à sa gauche ou à sa droite. C'est cette vision globale qui permet une compréhension contextuelle si riche. [SOURCE: Livre p.18, Figure 1-21]

### Le token [CLS] : Le porte-parole de la phrase
Dans un modèle de représentation, nous ne voulons pas générer la suite du texte, nous voulons un "résumé mathématique" de ce que le texte raconte. Pour cela, BERT introduit une convention géniale : le token **[CLS]** (pour *Classification*).

⚠️ **Attention : erreur fréquente ici !** Le token `[CLS]` n'est pas un mot magique. C'est simplement un token spécial que l'on place systématiquement au tout début de chaque entrée. Parce que BERT utilise la *Self-Attention* totale, ce token `[CLS]` va absorber des informations de *tous* les autres tokens de la phrase. 
🔑 **La règle d'or :** À la sortie de la 12ème couche, le vecteur correspondant au token `[CLS]` est utilisé comme la représentation compressée de toute la séquence. C'est ce vecteur que nous brancherons sur un classifieur pour décider si une critique est positive ou négative. [SOURCE: Livre p.40, p.113]

### Apprendre par les trous : Le Masked Language Modeling (MLM)
Comment entraîne-t-on un modèle à "comprendre" sans lui donner de labels ? On utilise un jeu d'enfant : le texte à trous. C'est le **Masked Language Modeling**, illustré dans les **Figures 1-22 et 1-23** (p.19-20).

Le processus est fascinant :
1.  On prend une phrase normale (ex: "Le chat boit du lait").
2.  On cache aléatoirement 15% des mots avec un token spécial **[MASK]**.
3.  On demande au modèle de deviner ce qu'il y avait sous le masque.

Pour réussir, BERT est forcé d'apprendre la grammaire ("Le" suggère un nom masculin), la syntaxe, et surtout la sémantique ("boit" et "lait" sont fortement liés au mot caché "chat"). [SOURCE: Livre p.19-20, Figures 1-22, 1-23]

### L'évolution de la lignée : De RoBERTa à DeBERTa
Depuis la sortie de BERT par Google en 2018, la famille s'est agrandie. Regardez la **Figure 4-5 : Timeline des modèles BERT-like** (p.115). Chaque nouveau-venu a apporté une innovation majeure. [SOURCE: Livre p.115, Figure 4-5]

#### 1. RoBERTa (Facebook AI)
RoBERTa est la preuve que "plus c'est long, meilleur c'est". Les chercheurs ont repris l'architecture de BERT mais l'ont entraînée sur beaucoup plus de données, pendant plus longtemps, et en supprimant une tâche d'entraînement secondaire jugée inutile (la *Next Sentence Prediction*). 🔑 **Notez bien :** RoBERTa est souvent le choix par excellence pour la classification aujourd'hui car ses représentations sont plus robustes.

#### 2. DistilBERT (Hugging Face)
« Tout le monde n'a pas un supercalculateur dans son garage ! » DistilBERT utilise une technique appelée "distillation de connaissances". On prend un gros BERT (le professeur) et on entraîne un petit BERT (l'élève) à imiter ses réponses. 
*   **Résultat** : 40% plus petit, 60% plus rapide, tout en conservant 97% des performances. C'est le modèle idéal pour les applications mobiles ou les serveurs à faibles ressources. [SOURCE: Livre p.115]

#### 3. ALBERT et DeBERTa
*   **ALBERT** : Utilise des astuces mathématiques pour réduire le nombre de paramètres sans perdre en puissance.
*   **DeBERTa** : Introduit l'attention "désentrelacée" (disentangled attention). Il sépare le contenu du mot de sa position relative. C'est actuellement l'un des modèles les plus performants sur les benchmarks de compréhension de lecture.

### Tableau comparatif de la famille BERT

| Modèle | Innovation Clé | Cas d'usage idéal |
| :--- | :--- | :--- |
| **BERT** | Premier modèle bidirectionnel massif | Baseline historique, recherche générale |
| **RoBERTa** | Entraînement optimisé, plus de données | Classification haute performance |
| **DistilBERT** | Distillation (plus léger/rapide) | Inférence en temps réel, Edge computing |
| **DeBERTa** | Attention désentrelacée | Tâches de raisonnement complexe |

[SOURCE: Livre p.115-116]

### Éthique et Transparence : Le miroir bidirectionnel
⚠️ **Fermeté bienveillante** : « Mes chers étudiants, soyez vigilants. » 
Parce que BERT regarde la phrase dans les deux sens, il est extrêmement sensible au contexte. Mais ce contexte contient nos biais. Si BERT voit 100 000 fois "L'infirmière est entrée dans la pièce" et "Le médecin est entré dans la pièce", l'embedding du mot `[MASK]` dans "Le [MASK] est entré..." sera statistiquement biaisé vers le genre masculin ou féminin selon la profession.

🔑 **Je dois insister :** Ces modèles de représentation ne sont pas des arbitres de vérité. Ce sont des miroirs statistiques de nos écrits. Lorsque vous utilisez BERT pour filtrer des CV ou analyser des sentiments, vous devez auditer les biais que le modèle pourrait avoir "appris" durant son MLM. [SOURCE: Livre p.28]

« Vous avez maintenant les clés de la famille BERT. Vous comprenez que leur force réside dans leur capacité à "figer" le sens d'un texte dans un vecteur riche et bidirectionnel. Dans la section suivante, nous verrons comment transformer ces connaissances théoriques en stratégies concrètes d'utilisation : faut-il tout réentraîner ou simplement "geler" le modèle ? »

---
*Fin de la section 4.1 (1150 mots environ)*
## 4.2 Stratégies d'utilisation (1000+ mots)

### L'athlète polyvalent : Pourquoi ne pas repartir de zéro ?
« Bonjour à nouveau ! Maintenant que vous connaissez la généalogie de BERT, une question cruciale se pose pour vous, ingénieurs et chercheurs : comment allons-nous utiliser ce géant ? Allez-vous passer six mois et dépenser des milliers d'euros pour entraîner votre propre modèle ? 🔑 **Je dois insister : la réponse est presque toujours NON.** »

Dans le monde des LLM, nous pratiquons ce que l'on appelle le **Transfer Learning** (Apprentissage par transfert). L'idée est simple mais révolutionnaire : nous prenons un modèle qui a déjà passé des mois à lire Wikipédia et des livres (le "Foundation Model") et nous allons l'adapter à notre petit problème spécifique, comme classer des tickets de support technique ou des diagnostics médicaux. Comme vous pouvez le voir sur la **Figure 4-3 : Fine-tuning d'un modèle foundation** (p.114), nous passons d'une connaissance générale à une expertise pointue. [SOURCE: Livre p.114, Figure 4-3]

### Stratégie 1 : Le Fine-tuning complet (Unfreezing)
C'est la méthode la plus puissante, mais aussi la plus gourmande. Imaginez que vous engagiez un chercheur brillant et que vous lui permettiez de remettre en question tout ce qu'il sait pour s'adapter à votre entreprise.
*   **Le principe** : On prend BERT, on ajoute une petite couche de classification à la fin, et on ré-entraîne *l'intégralité* des paramètres sur nos données.
*   **Quand l'utiliser ?** : Lorsque vous avez beaucoup de données étiquetées (plusieurs milliers) et que votre domaine est très différent du langage courant (par exemple, de la physique nucléaire ou du droit ancien).
*   **Inconvénient** : C'est lent et cela nécessite une carte graphique robuste (comme la T4 de notre Colab). ⚠️ **Attention : erreur fréquente ici !** Si vous ré-entraînez trop fort sur un petit jeu de données, vous risquez le "Catastrophic Forgetting" : le modèle devient un génie pour votre tâche mais oublie tout le reste du langage. [SOURCE: Livre p.114]

### Stratégie 2 : L'extraction de caractéristiques (Frozen Layers)
C'est ici que l'ingénierie devient élégante et accessible aux "GPU-poor". Regardez la **Figure 4-4 : Classification directe vs. indirecte** (p.114). Au lieu de modifier BERT, on le considère comme un dictionnaire immuable. 
*   **Le principe** : On "gèle" (freeze) les 12 couches de BERT. On lui donne une phrase, il nous rend un vecteur (l'embedding du token `[CLS]`), et nous entraînons un classifieur très simple (comme une régression logistique avec Scikit-Learn) par-dessus.
*   **Analogie** : C'est comme si vous utilisiez un expert pour traduire un texte complexe, puis que vous preniez sa traduction pour décider si le sujet est intéressant. Vous ne changez pas la façon dont l'expert travaille ; vous utilisez simplement son résultat. 
*   **Avantage** : C'est incroyablement rapide. Vous pouvez classer des millions de documents en quelques minutes car BERT ne fait qu'une passe de calcul sans jamais se mettre à jour. [SOURCE: Livre p.114, Figure 4-4]

### Modèle de sélection : Comment choisir son champion ?
« Mes chers étudiants, ne vous perdez pas dans les 60 000 modèles de Hugging Face ! » Pour réussir votre projet, votre sélection doit suivre trois critères non-négociables :
1.  **La langue** : N'utilisez pas un BERT entraîné uniquement sur l'anglais pour analyser du français. Cherchez des modèles comme `CamemBERT` (français) ou des modèles multilingues comme `mBERT` ou `XLM-RoBERTa`.
2.  **La taille** : Si vous déployez sur un téléphone, `DistilBERT` est votre meilleur ami. Si vous cherchez la précision absolue, `DeBERTa-v3` est le roi actuel.
3.  **Le domaine** : Il existe des versions spécialisées comme `SciBERT` (articles scientifiques) ou `BioBERT` (médecine). Utiliser un modèle déjà pré-adapté à votre domaine vous fera gagner des semaines de travail. [SOURCE: Livre p.115]

### Le baromètre mondial : MTEB Leaderboard
🔑 **Je dois insister :** Si vous voulez savoir quel modèle produit les meilleurs embeddings au monde aujourd'hui, vous devez consulter le **MTEB (Massive Text Embedding Benchmark) Leaderboard**. C'est le "Billboard Hot 100" de l'IA de représentation. Il classe les modèles sur des dizaines de tâches différentes. 
⚠️ **Fermeté bienveillante** : Un modèle qui est numéro 1 pour la recherche de documents ne sera pas forcément le meilleur pour la classification de sentiments. Regardez les colonnes spécifiques à votre tâche ! [SOURCE: Livre p.120, Blog 'LLM Roadmap' de Maarten Grootendorst]

### Implémentation pratique : La puissance de `pipeline`
Hugging Face a créé une abstraction magnifique pour nous simplifier la vie. Voici comment implémenter un classifieur de pointe en 3 lignes de code.

```python
# Installation : pip install transformers
from transformers import pipeline

# Choix d'un modèle RoBERTa optimisé pour le sentiment
# [SOURCE: CONCEPT À SOURCER – HF MODEL HUB cardiffnlp]
model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Création de la pipeline
# device=0 utilise le GPU T4 de Colab (indispensable pour la vitesse)
pipe = pipeline("sentiment-analysis", model=model_path, device=0)

# Test sur une critique de film
result = pipe("This film is a masterpiece of modern representation models!")

print(f"Résultat : {result}")
# Sortie attendue : [{'label': 'positive', 'score': 0.998}]
```

🔑 **Note technique** : Dans ce code, la pipeline télécharge automatiquement le tokeniseur et les poids du modèle, s'occupe du token `[CLS]` en coulisses et vous rend directement une étiquette humaine. C'est l'outil parfait pour prototyper rapidement.

### Éthique et Responsabilité : Le coût de l'expertise
⚠️ **Éthique ancrée** : « Mesurez l'impact environnemental de vos choix. » 
Le fine-tuning complet (Stratégie 1) consomme beaucoup d'énergie car vous recalculez les gradients pour des centaines de millions de paramètres à chaque itération. 
🔑 **Mon conseil de professeur** : Commencez TOUJOURS par la Stratégie 2 (modèle gelé). Si les performances ne sont pas suffisantes, passez alors au fine-tuning. Une IA responsable est aussi une IA sobre qui n'utilise pas un marteau-pilon pour écraser une mouche. [SOURCE: Livre p.28]

« Vous maîtrisez maintenant les stratégies. Vous savez quand geler vos couches et quand laisser le modèle apprendre. Dans la section suivante, nous allons voir comment évaluer ces modèles : comment savoir si notre BERT est vraiment devenu un expert ou s'il fait simplement semblant ? Nous parlerons de matrices de confusion et de scores F1. »

---
*Fin de la section 4.2 (1070 mots environ)*
## 4.3 Applications pratiques (1200+ mots)

### Passer de la théorie au terrain : Le métier de l'artisan IA
« Bonjour à toutes et à tous ! Nous entrons maintenant dans la phase que je préfère : celle où la théorie rencontre la réalité du terrain. Jusqu'ici, nous avons étudié BERT comme un objet fascinant en laboratoire. Mais dans la vie d'un ingénieur ou d'un chercheur, BERT est un outil, un pinceau ou un scalpel. Aujourd'hui, nous allons apprendre à l'utiliser pour résoudre un problème concret et, surtout, à mesurer si notre travail est de qualité. 🔑 **Je dois insister : un modèle sans métriques de performance n'est qu'une intuition coûteuse.** »

Pour cette démonstration, nous allons nous attaquer à un classique : l'analyse de sentiments sur le jeu de données **Rotten Tomatoes**. Ce corpus contient des milliers de critiques de films classées comme "positives" ou "négatives". C'est un terrain de jeu idéal car il est parfaitement équilibré (autant de positifs que de négatifs), ce qui nous facilitera l'interprétation initiale. [SOURCE: Livre p.112]

### Mise en œuvre : La recette du classifieur par embeddings
Comme nous l'avons vu en section 4.2, nous allons utiliser la stratégie de l'extraction de caractéristiques. Nous n'allons pas modifier les "connaissances" de notre modèle de langage, mais nous allons lui demander de nous extraire la substantifique moelle de chaque critique de film sous forme de vecteurs.

Le processus se déroule en trois étapes :
1.  **L'encodage** : Nous passons chaque texte dans un modèle de type `sentence-transformers` (qui est une version optimisée de BERT pour les phrases). Chaque critique devient un point dans un espace à 768 dimensions.
2.  **L'entraînement** : Nous utilisons ces 768 nombres comme caractéristiques (features) pour un classifieur simple. Ici, nous choisissons la **Régression Logistique**. Pourquoi ? Parce qu'elle est extrêmement rapide, stable et très efficace sur de petits volumes de données.
3.  **L'inférence** : Nous testons notre classifieur sur des données qu'il n'a jamais vues pour vérifier s'il a vraiment "compris" la notion de sentiment ou s'il a simplement mémorisé les exemples. [SOURCE: Livre p.114, p.121]

### Laboratoire de code : Classification de sentiments (Colab T4)
Voici l'implémentation complète. Remarquez la vitesse d'exécution : une fois les embeddings extraits, l'entraînement du classifieur prend moins d'une seconde !

```python
# Installation des dépendances
# !pip install sentence-transformers datasets scikit-learn

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# 1. Chargement du dataset
dataset = load_dataset("rotten_tomatoes")

# 2. Chargement du modèle de représentation (Frozen BERT-like)
# [SOURCE: CONCEPT À SOURCER – Modèle all-mpnet-base-v2 recommandé p.140]
model = SentenceTransformer("all-mpnet-base-v2", device="cuda")

# 3. Extraction des embeddings (On transforme le texte en nombres)
print("Extraction des caractéristiques en cours...")
X_train = model.encode(dataset["train"]["text"], show_progress_bar=True)
X_test = model.encode(dataset["test"]["text"], show_progress_bar=True)
y_train = dataset["train"]["label"]
y_test = dataset["test"]["label"]

# 4. Entraînement du classifieur léger (Scikit-Learn)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# 5. Évaluation
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["Négatif", "Positif"]))

# [SOURCE: CONCEPT À SOURCER – INSPIRÉ DU REPO GITHUB CHAPTER 4]
```

### Le juge de paix : La Matrice de Confusion
Une fois que le code a tourné, vous obtenez des chiffres. Mais comment les lire ? Regardez la **Figure 4-8 : Matrice de confusion** (p.119 du livre). C'est votre tableau de bord absolu. [SOURCE: Livre p.119, Figure 4-8]

Elle croise la réalité (*Actual values*) et la prédiction du modèle (*Predicted values*). Elle définit quatre destins pour une donnée :
1.  **Vrais Positifs (TP)** : Le film est génial, et BERT l'a trouvé génial. Bravo !
2.  **Vrais Négatifs (TN)** : Le film est un navet, et BERT a confirmé que c'était un navet.
3.  **Faux Positifs (FP)** : C'est l'erreur "optimiste". Le film est mauvais, mais le modèle l'a classé comme bon.
4.  **Faux Négatifs (FN)** : C'est l'erreur "pessimiste". Le film est un chef-d'œuvre, mais le modèle l'a détesté.

🔑 **Je dois insister :** Ne regardez jamais uniquement le score global (l'Accuracy). Si vous travaillez sur la détection de fraude et que 99% des transactions sont honnêtes, un modèle qui dit "Toujours Honnête" aura 99% d'accuracy mais sera totalement inutile pour attraper le 1% de voleurs. C'est là qu'interviennent nos trois piliers.

### Précision, Rappel et Score F1 : La Sainte Trinité
Comme l'illustre la **Figure 4-9 : Rapport de classification** (p.120), nous devons jongler avec trois mesures. [SOURCE: Livre p.120, Figure 4-9]

#### 1. La Précision (Precision)
C'est la question de la **fiabilité**. "Quand mon modèle dit que c'est positif, quelle est la probabilité qu'il ait raison ?"
*   *Analogie* : C'est comme un témoin au tribunal. On veut qu'il ne dise que la vérité. S'il accuse quelqu'un à tort (Faux Positif), sa précision chute.

#### 2. Le Rappel (Recall)
C'est la question de la **complétude**. "Sur tous les films positifs qui existent dans mon test, combien mon modèle a-t-il réussi à en attraper ?"
*   *Analogie* : C'est comme un filet de pêche. On veut qu'il attrape tous les poissons. Si des poissons s'échappent (Faux Négatifs), le rappel chute.

#### 3. Le Score F1
⚠️ **Attention : erreur fréquente ici !** On ne fait pas une simple moyenne arithmétique entre Précision et Rappel. On utilise une **moyenne harmonique**. Pourquoi ? Parce que la moyenne harmonique punit sévèrement les déséquilibres. Si votre précision est de 1.0 (parfaite) mais votre rappel de 0.0 (vous n'avez rien trouvé), votre score F1 sera de 0, et non de 0.5. C'est l'indicateur de la robustesse globale de votre application. [SOURCE: Livre p.120]

### Au-delà du sentiment : La Reconnaissance d'Entités Nommées (NER)
La classification de texte n'est que la partie émergée de l'iceberg. Une autre application majeure des modèles *Encoder-only* est le **NER** (*Named Entity Recognition*).
Ici, nous ne classons pas toute la phrase, mais chaque mot individuellement. 
*   "Elon Musk habite au Texas."
*   BERT va classer "Elon Musk" comme `PERSONNE` et "Texas" comme `LIEU`.

🔑 **La distinction technique :** Pour le sentiment, nous utilisions le vecteur du token `[CLS]`. Pour le NER, nous utilisons les vecteurs de *chaque* mot à la sortie du Transformer pour décider de leur étiquette. C'est la différence entre une vision globale et une vision granulaire. [SOURCE: Livre p.133]

### Éthique et Responsabilité : Le danger du "Label Noise"
⚠️ **Fermeté bienveillante** : « Mes chers étudiants, soyez critiques envers vos données. » 
Dans notre application pratique, nous supposons que les labels "positif" et "négatif" sont la vérité absolue. Mais l'humain est complexe ! Une critique peut être sarcastique : "Quel chef-d'œuvre d'ennui !". Un labelleur humain fatigué pourrait classer cela en positif à cause du mot "chef-d'œuvre". 

🔑 **Conséquence éthique :** Si vos données de départ sont mal étiquetées (le *Label Noise*), votre modèle va apprendre à reproduire ces erreurs. Avant de lancer un modèle BERT en production, passez du temps à regarder les cas où votre modèle se trompe. Souvent, vous découvrirez que c'est l'étiquette humaine qui était erronée ou ambiguë. L'IA n'est que le reflet de notre propre clarté. [SOURCE: Livre p.28]

« Vous avez maintenant les mains dans le code et les yeux sur les métriques. Vous savez transformer du texte en vecteurs, entraîner un cerveau de classification et juger sa performance. Dans la section suivante, nous allons corser le jeu : que se passe-t-il si nous n'avons AUCUN label ? Bienvenue dans le monde du Zero-shot et de la similarité cosinus. »

---
*Fin de la section 4.3 (1260 mots environ)*
## 4.4 Tâches de classification avancées (900+ mots)

### L'IA sans professeur : Le défi du monde réel
« Bonjour à toutes et à tous ! Imaginez un instant que vous soyez parachuté dans une entreprise qui reçoit des milliers de retours clients chaque jour. Votre patron vous demande de les classer par urgence, mais il y a un problème de taille : vous n'avez aucune donnée déjà étiquetée. Pas un seul exemple "urgent" ou "normal" pour entraîner votre classifieur Scikit-Learn de la section précédente. Allez-vous passer vos nuits à étiqueter à la main ? 🔑 **Je dois insister : avant de sortir vos étiqueteuses, sortez vos embeddings !** »

Dans cette dernière section de la semaine, nous allons explorer la magie de la classification **Zero-shot** (zéro-exemple). C'est l'une des applications les plus puissantes des modèles de représentation. Nous allons apprendre à classifier du texte simplement en utilisant la géométrie de l'espace vectoriel. Respirez, nous allons voir comment transformer un nom de classe en une position GPS mathématique. [SOURCE: Livre p.123]

### L'intuition du Zero-shot : La comparaison sémantique
La classification classique (supervisée) demande au modèle d'apprendre la frontière entre deux nuages de points. La classification Zero-shot, elle, repose sur une idée brillante illustrée dans les **Figures 4-13 à 4-16** (p.124-126 du livre). [SOURCE: Livre p.124-126, Figures 4-13 à 4-16]

Le processus est le suivant :
1.  **Embedder le document** : Nous transformons la phrase du client en un vecteur avec BERT (ex: "Mon colis est arrivé cassé !").
2.  **Embedder les étiquettes** : C'est le coup de génie. Nous transformons les noms des classes eux-mêmes en vecteurs. Nous créons un vecteur pour le mot "Urgent" et un vecteur pour le mot "Normal".
3.  **Comparer les distances** : Le document appartient à la classe dont le vecteur est le plus "proche" du sien dans l'espace multidimensionnel. 

🔑 **La distinction non-négociable :** Dans ce scénario, nous ne classons pas par rapport à une règle apprise, mais par rapport à la proximité sémantique brute. C'est le modèle de langage qui "sait" déjà que "cassé" est sémantiquement plus proche d' "Urgent" que de "Normal". [SOURCE: Livre p.124]

### L'outil de mesure : La Similarité Cosinus
Comment la machine calcule-t-elle cette "proximité" ? Elle utilise généralement la **Similarité Cosinus**, représentée en **Figure 4-15** (p.125). 

⚠️ **Attention : erreur fréquente ici !** On pourrait être tenté d'utiliser la distance euclidienne (la règle droite entre deux points). Mais en NLP, la longueur des vecteurs peut varier selon la richesse du texte. La similarité cosinus, elle, ne regarde que l'**angle** entre les vecteurs.
*   Si l'angle est de 0°, le cosinus est de 1 : les textes sont sémantiquement identiques.
*   Si l'angle est de 90°, le cosinus est de 0 : ils n'ont aucun rapport.
*   Si l'angle est de 180°, le cosinus est de -1 : ils sont opposés.

**Analogie** : C'est comme deux boussoles. Peu importe si une aiguille est plus longue que l'autre ; si elles pointent toutes les deux vers le Nord, elles sont "similaires". [SOURCE: Livre p.125, Figure 4-15]

### Implémentation : Zero-shot avec Scikit-Learn
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

# [SOURCE: CONCEPT À SOURCER – INSPIRÉ DU LIVRE p.125 ET DU REPO GITHUB CHAPTER 4]
```

### L'art de nommer les classes : Le "Label Prompting"
🔑 **Je dois insister :** La performance du Zero-shot dépend énormément de la façon dont vous nommez vos classes. 
Si vous nommez une classe "X12", BERT ne pourra rien faire. Si vous la nommez "Problème de facturation", il sera très efficace. 

Parfois, il est même préférable d'utiliser une petite description plutôt qu'un seul mot. Au lieu de "Sport", essayez "Un article traitant de compétitions sportives, d'athlètes ou de résultats de matchs". En enrichissant le vecteur de la classe, vous donnez plus de chances au document de s'y "aimanter". [SOURCE: Livre p.126]

### Limites et Bonnes Pratiques
⚠️ **Fermeté bienveillante** : « Ne tombez pas dans la paresse technologique. » Le Zero-shot a des limites claires :
1.  **Sensibilité au vocabulaire** : Si votre domaine utilise un jargon très spécifique que BERT n'a pas vu (ex: des codes d'erreur industriels), la similarité sera faible.
2.  **Performance** : Un modèle entraîné avec des labels (supervisé) sera TOUJOURS plus précis qu'un modèle Zero-shot. Le Zero-shot est une excellente solution de secours ou un point de départ pour pré-étiqueter vos données.
3.  **Coût de calcul** : Si vous avez 1000 classes, vous devez comparer chaque document à 1000 vecteurs de labels, ce qui peut ralentir le système.

### Éthique et Biais dans le choix des labels
⚠️ **Éthique ancrée** : « Mes chers étudiants, le choix des mots n'est jamais neutre. » 
En Zero-shot, c'est **vous** qui définissez l'espace sémantique. Si vous créez une classe "Sentiment agressif" au lieu de "Réclamation client", vous orientez déjà la façon dont l'IA va percevoir vos utilisateurs. 
🔑 **Conséquence éthique :** Les biais du modèle (vus en 4.1) vont interagir avec vos labels. Si le modèle associe statistiquement certains dialectes ou manières de parler à une classe négative, vous risquez de discriminer certains groupes sans même avoir entraîné le modèle. Auditez toujours les résultats du Zero-shot pour vérifier qu'une catégorie de population n'est pas systématiquement mal classée par pur préjugé statistique du modèle. [SOURCE: Livre p.28]

« Nous avons bouclé la boucle ! De la structure rigide de BERT aux nuances fluides du Zero-shot, vous maîtrisez maintenant la science des modèles de représentation. Vous savez non seulement comment ils fonctionnent, mais comment les déployer quand les données manquent. La semaine prochaine, nous basculerons du côté de la création avec les modèles de génération (GPT). Mais avant cela, rendez-vous au laboratoire pour mettre tout cela en pratique ! »

---
*Fin de la section 4.4 (980 mots environ)*
## 🧪 LABORATOIRE SEMAINE 4 (750+ mots)

**Accroche du Professeur Khadidja Henni** : 
« Bonjour à nouveau ! J'espère que vous avez les yeux bien ouverts, car nous passons à la pratique. Ce laboratoire est le moment de vérité : nous allons transformer nos concepts abstraits de BERT et de vecteurs en outils concrets. 🔑 **Je dois insister :** ne vous contentez pas de copier le code. Regardez comment chaque ligne influence le résultat. Nous allons voir comment un modèle "frozen" peut devenir un allié puissant pour classer des données en un temps record. Prêt·e·s ? Sortez vos notebooks Colab ! »

---

### 🔹 QUIZ MCQ (10 questions)

1. **Quel token spécial est utilisé par BERT pour représenter l'intégralité d'une phrase en vue d'une classification ?**
   a) `[SEP]`
   b) `[MASK]`
   c) `[CLS]`
   d) `[PAD]`
   **[Réponse: c]** [Explication: Le token `[CLS]` (Classification) accumule les informations de toute la séquence grâce à la self-attention bidirectionnelle. SOURCE: Livre p.18, p.113]

2. **Quelle technique d'entraînement permet à BERT d'apprendre des représentations bidirectionnelles sans labels humains ?**
   a) La descente de gradient stochastique
   b) Le Masked Language Modeling (MLM)
   c) La prédiction du mot suivant (Next Token Prediction)
   d) Le clustering K-means
   **[Réponse: b]** [Explication: En cachant 15% des mots, le modèle apprend à reconstruire le sens à partir du contexte gauche et droit. SOURCE: Livre p.19-20]

3. **Quel modèle est une version distillée, 40% plus petite et 60% plus rapide que BERT ?**
   a) RoBERTa
   b) DeBERTa
   c) DistilBERT
   d) GPT-2
   **[Réponse: c]** [Explication: DistilBERT utilise la distillation de connaissances pour conserver 97% des performances de BERT dans un format compact. SOURCE: Livre p.115]

4. **Quelle métrique de classification est la plus robuste lorsqu'on cherche un équilibre entre la précision et le rappel ?**
   a) L'Accuracy globale
   b) La perte (Loss)
   c) Le score F1
   d) Le taux d'erreur
   **[Réponse: c]** [Explication: Le score F1 est la moyenne harmonique de la précision et du rappel, punissant les déséquilibres entre les deux. SOURCE: Livre p.120]

5. **Dans un modèle task-specific, quel composant convertit les embeddings de BERT en prédictions de classes finales ?**
   a) La couche d'attention
   b) La tête de classification (couche linéaire/dense)
   c) Le tokeniseur
   d) La couche d'activation ReLU
   **[Réponse: b]** [Explication: On ajoute une couche "Head" par-dessus le Transformer pour projeter le vecteur `[CLS]` vers le nombre de classes souhaité. SOURCE: Livre p.113, Figure 4-3]

6. **Quelle approche permet de classer du texte dans des catégories sans avoir besoin d'un jeu de données d'entraînement étiqueté ?**
   a) Le Supervised Fine-tuning
   b) La classification Zero-shot
   c) Le Bag-of-Words
   d) La régression logistique
   **[Réponse: b]** [Explication: Elle compare la similarité sémantique entre le texte et les étiquettes transformées en vecteurs. SOURCE: Livre p.123]

7. **Combien de couches (Transformer blocks) possède la version "base" de BERT ?**
   a) 6
   b) 12
   c) 24
   d) 48
   **[Réponse: b]** [Explication: BERT-base est constitué de 12 blocs d'encodeurs empilés. SOURCE: Livre p.18, Figure 1-21]

8. **Quel est l'avantage principal du fine-tuning (réglage fin) par rapport à l'utilisation d'un modèle gelé (frozen) ?**
   a) C'est beaucoup plus rapide
   b) Cela nécessite moins de mémoire
   c) Cela permet au modèle d'adapter ses représentations internes au vocabulaire spécifique de votre domaine
   d) Cela ne nécessite aucun label
   **[Réponse: c]** [Explication: En dégelant les couches, les poids de BERT s'ajustent pour mieux "comprendre" les nuances spécifiques de vos données. SOURCE: Livre p.114]

9. **Quelle technique mathématique est la plus utilisée pour mesurer la similarité entre deux embeddings textuels ?**
   a) La distance de Manhattan
   b) La similarité Cosinus
   c) L'écart type
   d) La multiplication simple
   **[Réponse: b]** [Explication: Elle mesure l'angle entre les vecteurs, ignorant leur longueur pour se concentrer sur l'orientation sémantique. SOURCE: Livre p.125, Figure 4-15]

10. **Quel modèle récent a introduit l'attention "désentrelacée" (disentangled attention) pour surpasser BERT ?**
    a) ALBERT
    b) DeBERTa
    c) CamemBERT
    d) RoBERTa
    **[Réponse: b]** [Explication: DeBERTa sépare les informations de contenu et de position pour un raisonnement plus fin. SOURCE: Livre p.115]

---

### 🔹 EXERCICE 1 : Classification de sentiments avec Pipeline (Niveau Basique)

**Objectif** : Apprendre à charger et utiliser une pipeline Hugging Face sur le GPU T4 de Colab pour une classification immédiate.

```python
# --- CODE FOURNI (QUESTION) ---
from transformers import pipeline
from datasets import load_dataset

# Chargement d'un échantillon de test de Rotten Tomatoes
test_data = load_dataset("rotten_tomatoes", split="test").select(range(3))

# --- VOTRE TÂCHE : Initialisez la pipeline et lancez l'inférence ---

# --- RÉPONSE COMPLÈTE (CORRIGÉ) ---
# [SOURCE: CONCEPT À SOURCER – HF cardiffnlp p.116]
# Initialisation de la pipeline avec un modèle RoBERTa spécialisé
pipe = pipeline("sentiment-analysis", 
                model="cardiffnlp/twitter-roberta-base-sentiment-latest", 
                device=0) # device=0 active le GPU T4 (Crucial pour la vitesse !)

# Boucle d'inférence
for text in test_data["text"]:
    # Appel de la pipeline sur le texte
    result = pipe(text)
    print(f"Texte: {text[:60]}...")
    print(f"Prédiction: {result[0]['label']} (Score: {result[0]['score']:.4f})\n")
```
**Attentes** : Vous devez observer que le modèle identifie correctement le sentiment. ⚠️ **Avertissement du Professeur** : Notez bien que le score représente la confiance du modèle. Un score de 0.51 signifie que BERT est presque aussi confus que nous devant une critique sarcastique !

---

### 🔹 EXERCICE 2 : Embeddings + Classifieur Scikit-Learn (Niveau Intermédiaire)

**Objectif** : Implémenter la Stratégie 2 (Frozen Layers) en extrayant des caractéristiques avec `sentence-transformers` et en entraînant un modèle linéaire.

```python
# --- CODE FOURNI (QUESTION) ---
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from datasets import load_dataset
import numpy as np

# Charger 200 exemples pour l'entraînement
dataset = load_dataset("rotten_tomatoes")
train_subset = dataset["train"].select(range(200))
model = SentenceTransformer("all-mpnet-base-v2")

# --- VOTRE TÂCHE : Extraire les embeddings et entraîner la LogisticRegression ---

# --- RÉPONSE COMPLÈTE (CORRIGÉ) ---
# [SOURCE: CONCEPT À SOURCER – Stratégie Frozen p.121]
# 1. Extraction des caractéristiques (On transforme le texte en vecteurs)
X_train = model.encode(train_subset["text"])
y_train = train_subset["label"]

# 2. Initialisation et entraînement du classifieur (Régression Logistique)
clf = LogisticRegression()
clf.fit(X_train, y_train)

# 3. Vérification de la performance sur le train set
score = clf.score(X_train, y_train)
print(f"Précision sur l'échantillon d'entraînement : {score*100:.2f}%")
```
**Attentes** : Comprendre pourquoi l'entraînement est quasi instantané. 🔑 **Je dois insister :** BERT a déjà fait tout le travail de compréhension du langage pendant son pré-entraînement ; notre classifieur ne fait que tracer une ligne entre les points. [SOURCE: Livre p.121]

---

### 🔹 EXERCICE 3 : Logique de classification Zero-shot (Niveau Avancé)

**Objectif** : Implémenter manuellement la classification Zero-shot en utilisant la similarité cosinus entre un document et des étiquettes textuelles.

```python
# --- CODE FOURNI (QUESTION) ---
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer("all-mpnet-base-v2")
doc = "I am so annoyed by the constant software crashes!"
candidate_labels = ["technical issue", "billing question", "general praise"]

# --- VOTRE TÂCHE : Trouvez l'étiquette la plus proche via similarité cosinus ---

# --- RÉPONSE COMPLÈTE (CORRIGÉ) ---
# [SOURCE: CONCEPT À SOURCER – Zero-shot similarity p.125]
# 1. Encodage du document et des étiquettes (labels)
doc_embedding = model.encode([doc])
label_embeddings = model.encode(candidate_labels)

# 2. Calcul de la similarité cosinus (produit des angles)
# Cela nous donne un score pour chaque label
similarities = cosine_similarity(doc_embedding, label_embeddings)

# 3. Identification de l'index du score le plus élevé (argmax)
best_index = np.argmax(similarities)

print(f"Document : '{doc}'")
print(f"Classe prédite (Zero-shot) : {candidate_labels[best_index]}")
print(f"Score de similarité : {similarities[0][best_index]:.4f}")
```
**Attentes** : Expliquez l'impact du changement de label. ⚠️ **Fermeté bienveillante :** Si vous remplacez "technical issue" par "computer problem", le score changera. Le Zero-shot est un art de la précision sémantique ! [SOURCE: Livre p.126]

---

**Mots-clés de la semaine** : BERT, Encodeur, Bidirectionnel, MLM, Token [CLS], Fine-tuning, Représentation creuse vs dense, Score F1, Similarité Cosinus, Zero-shot.

**En prévision de la semaine suivante** : Nous allons changer de paradigme. Finie la compréhension pure, place à la création ! Nous explorerons les modèles **Decoder-only** (famille GPT) et l'art de murmurer à l'oreille des IA : le *Prompt Engineering*. [SOURCE: CSV Transition]

**SOURCES COMPLÈTES** :
*   Livre : Alammar & Grootendorst (2024), *Hands-On LLMs*, Chapitre 4, p.111-135.
*   Hugging Face Blog : *Sentence Transformers* (https://huggingface.co/blog/sentence-transformers).
*   SBERT Documentation : https://www.sbert.net/
*   GitHub Officiel : chapter04 repository.

[/CONTENU SEMAINE 4]
