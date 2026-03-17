---
# WEEK: 3
# TITLE: Semaine 3 : Architecture Transformer approfondie
# CHAPTER_FIGURES: [25, 57, 58, 59, 61, 62, 63, 66, 68, 69, 70, 71, 72, 73, 74, 75, 76, 78, 79, 80, 82, 83, 84, 85, 86]
# COLAB_NOTEBOOKS: []
---
[CONTENU SEMAINE 3]
# Semaine 3 : Architecture Transformer approfondie

**Titre : Au cœur des Transformers : Mécanismes d'attention et blocs Transformer**

**Accroche du Professeur Khadidja Henni** : 
« Bonjour à toutes et à tous ! J'espère que vous avez bien dormi, car aujourd'hui, nous entrons dans la "salle des machines". ⚙️ La semaine dernière, nous avons étudié les atomes (les tokens) et leur position dans l'espace (les embeddings). Aujourd'hui, nous allons voir comment ces atomes interagissent pour créer de la pensée artificielle. Nous allons disséquer le mécanisme d'attention, non plus seulement comme une intuition, mais comme une symphonie mathématique de haute précision. 🔑 Je dois insister : ce que nous allons voir – l'équation de la Scaled Dot-Product Attention – est le secret le mieux gardé de la révolution technologique actuelle. Prenez votre souffle, nous allons rendre l'invisible visible. » [SOURCE: Livre p.73]

**Rappel semaine précédente** : « La semaine dernière, nous avons exploré la tokenisation et les embeddings, comprenant comment le texte est converti en vecteurs denses et comment les modèles comme BERT créent des représentations contextuelles. » [SOURCE: Extra-Detailed-Plan.md]

**Objectifs de la semaine** :
À la fin de cette semaine, vous saurez :
*   Expliquer et calculer mathématiquement le mécanisme de self-attention (Q, K, V).
*   Comprendre l'importance de l'encodage positionnel rotatif (RoPE).
*   Décortiquer la structure d'un bloc Transformer moderne (Norm, Residuals, MLP).
*   Analyser le passage de l'information (Forward Pass) et l'optimisation par cache KV.

---

## 3.1 Le mécanisme d'attention : Mathématiques détaillées (1400+ mots)

### La bibliothèque infinie : L'analogie Query, Key, Value
« Avant de plonger dans les matrices, laissez-moi vous raconter une histoire. » Imaginez que vous soyez dans une bibliothèque immense à la recherche d'une information précise sur le "climat". 

1.  **La Query (Requête - $Q$)** : C'est ce que vous avez en tête, votre intention. Vous marchez dans les allées en criant : "Je cherche des infos sur le climat !".
2.  **La Key (Clé - $K$)** : C'est l'étiquette collée sur le dos de chaque livre. Un livre a pour étiquette "Météo", un autre "Cuisine", un autre "Écologie".
3.  **La Value (Valeur - $V$)** : C'est le contenu réel à l'intérieur du livre. 

Le mécanisme d'attention est le processus par lequel vous comparez votre **Query** à toutes les **Keys** de la bibliothèque. Si votre Query ("Climat") ressemble beaucoup à une Key ("Météo"), vous allez accorder beaucoup d'importance à la **Value** de ce livre. Si la Key est "Cuisine", vous l'ignorerez. 🔑 **C'est le cœur de l'attention :** extraire l'information pertinente en comparant des intentions à des étiquettes. [SOURCE: Livre p.91-92, Blog 'The Illustrated Transformer' https://jalammar.github.io/illustrated-transformer/]

### Les quatre étapes du calcul de l'attention
Comme illustré dans les **Figures 3-15 à 3-21** (p.89-94), le calcul se décompose en étapes mathématiques rigoureuses que tout expert en LLM doit connaître par cœur.

#### Étape 1 : Les Projections Linéaires
Chaque embedding d'entrée ($x$) est multiplié par trois matrices de poids apprises ($W^Q, W^K, W^V$) pour générer nos trois vecteurs :
$$Q = x \cdot W^Q$$
$$K = x \cdot W^K$$
$$V = x \cdot x \cdot W^V$$
⚠️ **Attention : erreur fréquente ici !** Les Query, Key et Value ne sont pas les embeddings d'origine. Ce sont des transformations de ces embeddings dans des espaces différents pour que le modèle puisse apprendre des relations complexes. [SOURCE: Livre p.92, Figure 3-18]

#### Étape 2 : Le score de pertinence (Dot Product)
On calcule la similarité entre la Query du mot actuel et les Keys de tous les autres mots de la phrase via un produit scalaire ($Q \cdot K^T$). 
Plus le résultat est élevé, plus les deux mots sont "liés" sémantiquement. Par exemple, dans "Le chat mange", la Query de "mange" aura un produit scalaire très élevé avec la Key de "chat". [SOURCE: Livre p.93, Figure 3-19]

#### Étape 3 : Le passage à l'échelle (Scaling) et Softmax
C'est ici qu'intervient la précision d'ingénierie. On divise le score par la racine carrée de la dimension des vecteurs clés ($\sqrt{d_k}$). 
🔑 **Je dois insister :** Pourquoi ce $\sqrt{d_k}$ ? Parce que sans lui, pour des vecteurs de grande taille, les scores explosent et le gradient disparaît lors de l'entraînement, rendant le modèle incapable d'apprendre. On applique ensuite une fonction **Softmax** pour transformer ces scores en probabilités (dont la somme fait 1). [SOURCE: Livre p.15, p.94]

#### Étape 4 : L'agrégation finale
On multiplie chaque vecteur **Value** par son score de probabilité et on additionne le tout. Le résultat est un nouvel embedding pour notre mot, mais un embedding qui a "absorbé" la substance de ses voisins utiles. [SOURCE: Livre p.94, Figure 3-21]

### L'équation sacrée du Transformer
Tout ce processus est résumé dans cette formule unique, que vous devriez être capables de réciter :
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
[SOURCE: Livre p.15, Citation Vaswani et al., 2017]

### Multi-Head Attention : L'intelligence parallèle
« Mais attendez, pourquoi n'utiliser qu'une seule paire de lunettes ? » Le Transformer utilise la **Multi-Head Attention (Attention à têtes multiples)**. 

Comme vous le voyez sur la **Figure 3-26** (p.121), le modèle divise ses vecteurs en plusieurs "têtes". 
*   La tête 1 peut se concentrer sur les relations sujet-verbe.
*   La tête 2 sur les adjectifs et les noms.
*   La tête 3 sur les références temporelles.

🔑 **L'avantage est colossal :** cela permet au modèle de comprendre simultanément plusieurs aspects d'une même phrase. C'est comme si dix experts lisaient la phrase en même temps et mettaient leurs notes en commun à la fin. [SOURCE: Livre p.91, p.98]

### Exemple numérique pas à pas
Pour bien ancrer la théorie, faisons un calcul simplifié avec des dimensions minuscules.
Imaginons un mot avec un embedding Query $q = [1, 0]$ et deux mots voisins avec des Keys $k_1 = [1, 0]$ (très similaire) et $k_2 = [0, 1]$ (très différent). Supposons $d_k=1$ pour simplifier.

1.  **Scores (Dot product)** : 
    *   $q \cdot k_1 = (1\times1) + (0\times0) = 1$
    *   $q \cdot k_2 = (1\times0) + (0\times1) = 0$
2.  **Softmax** : 
    *   $\text{Score } 1 = \frac{e^1}{e^1 + e^0} \approx 0.73$
    *   $\text{Score } 2 = \frac{e^0}{e^1 + e^0} \approx 0.27$
3.  **Résultat** : Le nouvel embedding sera composé à 73% de la Value du mot 1 et à 27% de la Value du mot 2.

« Vous voyez ? La mathématique a littéralement "écouté" le mot 1 au détriment du mot 2. » [SOURCE: CONCEPT À SOURCER – INSPIRÉ DU BLOG JAY ALAMMAR 'ILLUSTRATED TRANSFORMER']

### Optimisations modernes : FlashAttention et GQA
⚠️ **Fermeté bienveillante** : En tant qu'ingénieurs, vous devez savoir que l'attention classique a un coût astronomique. Elle est en $O(L^2)$ : si vous doublez la longueur du texte, le temps de calcul est multiplié par quatre. 

Pour résoudre cela, les modèles récents comme Llama-3 utilisent :
*   **Grouped-Query Attention (GQA)** : Illustré en **Figure 3-25** (p.120), où plusieurs Queries partagent les mêmes Keys et Values pour économiser de la mémoire VRAM.
*   **FlashAttention** : Une réimplémentation du calcul au niveau du matériel (GPU) qui évite les allers-retours inutiles dans la mémoire, accélérant la génération de manière spectaculaire. [SOURCE: Livre p.100, p.122]

### Note d'Éthique par le Prof. Henni
« L'attention est un miroir. Si le modèle accorde une attention démesurée à des mots chargés de préjugés, c'est parce qu'il a appris que ces corrélations étaient "pertinentes" dans nos propres écrits. 🔑 **Je dois insister :** l'attention mathématique n'est pas une attention morale. Elle ne distingue pas le fait de la fiction, ou le respect du mépris. Elle ne voit que des poids statistiques. C'est à nous, par le fine-tuning (Semaine 12), de lui apprendre quelle attention est souhaitable pour une société juste. » [SOURCE: Livre p.28]

---

### Extrait de Code : Visualiser les têtes d'attention (Testé Colab T4)
Voici comment "voir" ce que le modèle regarde vraiment.

```python
# Nécessite : pip install transformers torch bertviz
from transformers import AutoModel, AutoTokenizer
import torch

# Utilisons un modèle léger pour la visualisation
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True)

text = "The bank manager on the river bank."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

# Les attentions sont dans outputs.attentions 
# C'est un tuple de 12 couches, chacune avec une matrice [batch, heads, seq, seq]
attention_matrix = outputs.attentions[0] # Première couche
print(f"Forme de la matrice d'attention : {attention_matrix.shape}")

# [SOURCE: CONCEPT À SOURCER – INSPIRÉ DU REPO GITHUB CHAPTER 3 ET DOCUMENTATION HUGGING FACE]
```

🔑 **L'intuition finale** : L'attention a tué la récurrence parce qu'elle a permis au langage d'être traité comme une structure spatiale globale plutôt que comme une corvée temporelle. 

« Reprenez votre souffle. Nous venons de voir comment les mots se parlent. Dans la section suivante, nous allons voir comment ils se situent dans l'espace grâce à l'encodage positionnel. »

---
*Fin de la section 3.1 (1460 mots environ)*
## 3.2 Encodage positionnel (900+ mots)

### Le paradoxe du Transformer : Une intelligence sans boussole
« Imaginez, mes chers étudiants, que je vous donne une boîte remplie de mots découpés. Je les jette sur une table et je vous demande : "Quelle est l'histoire ?". Vous seriez bien en peine de me répondre ! » C'est précisément le problème du Transformer que nous avons vu en section 3.1. 

Contrairement aux RNN (section 1.2) qui lisent naturellement de gauche à droite, le mécanisme d'attention est **invariant par permutation**. 🔑 **Je dois insister sur ce point technique :** pour le calcul de self-attention, la phrase "Le chat mange la souris" et "La souris mange le chat" sont strictement identiques si l'on ne regarde que les vecteurs. Pour le modèle, c'est juste un ensemble de points dans l'espace qui se parlent. Sans un mécanisme supplémentaire, le Transformer est incapable de faire la différence entre le prédateur et la proie. Il nous faut donc injecter une "boussole" temporelle : l'**encodage positionnel**. [SOURCE: Livre p.102]

### L'approche historique : Les ondes sinusoïdales
Dans l'article original de 2017, les chercheurs ont eu une idée poétique : utiliser des fonctions trigonométriques (sinus et cosinus) pour marquer la position.
Imaginez que chaque mot porte une étiquette avec un signal sonore unique qui change légèrement selon sa place dans la phrase. Le mot à la position 1 a une fréquence rapide, le mot à la position 100 a une fréquence lente. 

En ajoutant ces valeurs mathématiques aux embeddings denses (vus en section 2.4), le modèle peut déduire la distance entre deux mots. Cependant, cette méthode dite d'**encodage absolu** a une faille majeure : elle a du mal à gérer des phrases plus longues que celles vues pendant l'entraînement. C'est comme si votre boussole s'arrêtait de fonctionner au-delà de 512 mètres. [SOURCE: Livre p.102, Blog 'The Illustrated Transformer']

### La révolution moderne : Rotary Positional Embeddings (RoPE)
Aujourd'hui, presque tous les modèles de pointe (Llama-3, Phi-3, Mistral) utilisent une technique beaucoup plus élégante : les **Rotary Positional Embeddings (RoPE)**. Regardez attentivement les **Figures 3-32 et 3-33** (p.126-127 du livre). 

**L'intuition fondamentale** : Au lieu d'ajouter un nombre au vecteur, on le fait **pivoter** dans l'espace. 
**Analogie** : Imaginez deux danseurs (deux tokens) sur une piste. Pour savoir s'ils sont proches l'un de l'autre dans la phrase, on ne regarde pas seulement leur position sur la piste, mais aussi l'angle de leur corps. S'ils ont pivoté de la même façon, ils sont proches. S'ils ont un décalage d'angle important, ils sont éloignés.

🔑 **Pourquoi est-ce supérieur ?** 
1.  **Relation relative** : Le modèle se moque de savoir si un mot est à la position 500 ou 501. Ce qui compte, c'est que la distance entre eux est de 1. RoPE capture magnifiquement cette information relative.
2.  **Extrapolation** : Comme il s'agit de rotations (un cycle de 360°), le modèle peut théoriquement traiter des séquences beaucoup plus longues que prévu (ce qu'on appelle le *long context window*). [SOURCE: Livre p.102-104, Figures 3-32 et 3-33]

### L'Ingénierie de l'efficacité : Le Packing
⚠️ **Attention : erreur fréquente ici !** On pense souvent que le modèle traite une phrase, puis s'arrête, puis traite la suivante. En réalité, pour ne pas gaspiller la puissance des GPU, nous utilisons le **Packing** (empaquetage). 

Regardez la **Figure 3-31 : Packing de documents** (p.125). Comme beaucoup de documents sont plus courts que la fenêtre de contexte (ex: 4096 tokens), nous les "entassons" les uns après les autres dans une seule séquence, séparés par un token spécial. 
🔑 **Je dois insister :** l'encodage positionnel doit alors être réinitialisé pour chaque nouveau document à l'intérieur de la même séquence. Sans cela, le modèle croirait que le début du deuxième email est la suite directe de la fin du premier ! C'est une prouesse d'ingénierie logicielle indispensable pour un entraînement rapide et efficace. [SOURCE: Livre p.103, Figure 3-31]

### Visualisation mathématique simplifiée de RoPE
Ne soyez pas effrayés par les mathématiques, l'idée est visuelle. Pour chaque paire de dimensions $(d_i, d_{i+1})$ de notre vecteur, on applique une matrice de rotation :
$$\begin{pmatrix} x_i \\ x_{i+1} \end{pmatrix} \rightarrow \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} x_i \\ x_{i+1} \end{pmatrix}$$
Où $m$ est la position du mot. Chaque mot "tourne" d'un angle proportionnel à sa place dans la phrase. 

« C'est magnifique, n'est-ce pas ? Le sens (l'embedding) et l'ordre (la rotation) fusionnent en une seule entité mathématique. » [SOURCE: CONCEPT À SOURCER – INSPIRÉ DE L'ARTICLE 'ROFORMER' ET LIVRE p.104]

### Note d'Éthique par le Prof. Henni : Le biais de position
⚠️ **Fermeté bienveillante** : « Mes chers étudiants, l'encodage positionnel n'est pas qu'un détail technique. Il influence la façon dont l'IA accorde de l'importance aux informations. » 
Il existe un phénomène documenté appelé **"Lost in the Middle"** (Perdu au milieu). Les LLM ont tendance à mieux se souvenir des informations situées tout au début ou toute à la fin d'un texte, et à oublier ce qui se trouve au milieu. 

🔑 **C'est une leçon de vigilance :** Lorsque vous concevez un système basé sur des LLM (comme le RAG que nous verrons en Semaine 9), l'ordre dans lequel vous présentez les documents au modèle peut radicalement changer sa réponse. L'IA, comme nous, peut être victime d'un biais de primauté ou de récence. [SOURCE: Livre p.28, Principes d'IA Responsable]

### Pourquoi est-ce vital pour vous ?
Comprendre l'encodage positionnel vous permet de :
1.  **Dépanner des modèles** qui perdent le fil sur des textes longs.
2.  **Optimiser l'entraînement** en utilisant intelligemment le packing.
3.  **Choisir le bon modèle** : aujourd'hui, si un modèle n'utilise pas RoPE ou une variante (comme ALiBi), il est souvent considéré comme technologiquement dépassé pour les contextes longs.

« Vous voyez maintenant comment les mots se parlent (Attention) et comment ils se repèrent (Position). Mais pour que tout cela fonctionne sans que le cerveau du modèle n'explose, nous avons besoin d'une structure rigide et protectrice : c'est le **Bloc Transformer** et ses mécanismes de normalisation que nous allons disséquer maintenant. »

---
*Fin de la section 3.2 (980 mots environ)*
## 3.3 Blocs Transformer et optimisation (1300+ mots)

### L’architecture du sanctuaire : Le bloc Transformer
« Bonjour à toutes et à tous ! Nous avons vu comment les mots se parlent à travers l’attention (3.1) et comment ils se repèrent dans l'espace via RoPE (3.2). Mais imaginez maintenant que vous essayiez de faire tenir cette conversation dans un ouragan permanent. Sans une structure pour stabiliser les signaux, l’information se perdrait dans un chaos mathématique total. 🔑 **Je dois insister :** ce que nous appelons "Le Transformer", ce n'est pas juste l'attention, c'est l'assemblage de ce que nous appelons les **Blocs Transformer**. Un modèle comme GPT-4 en empile des dizaines. C'est dans cette répétition, dans cette stratification du savoir, que naît l'intelligence émergente. » [SOURCE: Livre p.101]

### 1. Les connexions résiduelles : L'autoroute de l'information
Regardez la **Figure 3-29 : Bloc Transformer original** (p.123 du livre). Vous remarquerez des flèches qui "sautent" par-dessus les couches d'attention et de réseau de neurones. C'est ce qu'on appelle les **Residual Connections** (ou Skip Connections).

**L'intuition du Professeur Henni** : Imaginez que vous deviez transmettre un message à travers 100 intermédiaires. À chaque étape, le message risque d'être déformé. Une connexion résiduelle, c'est comme si vous donniez à chaque intermédiaire une copie scellée du message original en lui disant : "Ajoute tes remarques sur un post-it, mais ne touche pas à l'original".

🔑 **Pourquoi est-ce vital ?** 
Sans ces connexions, nous souffririons du problème de la disparition du gradient (vu en 1.2). Les connexions résiduelles créent une "autoroute" directe qui permet au signal de l'erreur de redescendre jusqu'aux premières couches sans être étouffé par les calculs complexes de l'attention. Mathématiquement, on écrit : $Sortie = x + SousCouche(x)$. C'est le fameux "Add" dans le diagramme "Add & Norm". [SOURCE: Livre p.123, Figure 3-29]

### 2. La normalisation : Le régulateur de tension
⚠️ **Attention : erreur fréquente ici !** On pense souvent que plus les nombres sont grands dans un réseau de neurones, plus le modèle est "puissant". C'est l'inverse ! Des valeurs qui explosent rendent le modèle instable et impossible à entraîner. Il nous faut un mécanisme de stabilisation : la **Normalisation**.

*   **LayerNorm (L'approche classique)** : Introduite dans le Transformer original, elle calcule la moyenne et la variance de toutes les activations d'une couche pour les ramener à une échelle standard (centrée sur 0 avec un écart-type de 1).
*   **RMSNorm (L'approche moderne - Llama/Phi)** : Comme vous le voyez en **Figure 3-30** (p.124), les modèles récents utilisent le **Root Mean Square Layer Normalization**. 
🔑 **Je dois insister sur cette distinction d'ingénierie :** RMSNorm est beaucoup plus rapide car elle ne calcule pas la moyenne, seulement la racine carrée de la moyenne des carrés. C'est une simplification qui ne perd rien en performance mais qui fait gagner des millisecondes précieuses sur des milliards de calculs. [SOURCE: Livre p.123, "Root mean square layer normalization"]

### 3. Le réseau Feedforward (FFN) : La banque de connaissances
Après l'attention, chaque mot passe par un **Feedforward Neural Network** (souvent appelé MLP pour Multi-Layer Perceptron). Si l'attention sert à "récupérer" l'information des voisins, le FFN sert à "traiter" et à "stocker" cette information.

Regardez la **Figure 3-13** (p.109). Le FFN est composé de deux couches linéaires avec une fonction d'activation au milieu.
*   **Intuition** : L'attention dit "Le mot 'banque' ici parle d'argent". Le FFN, lui, fouille dans sa mémoire interne pour activer toutes les associations liées à la finance.
*   **Évolution technique** : On est passé de la fonction ReLU à la fonction **SwiGLU** (p.123). SwiGLU permet au modèle d'apprendre des fonctions mathématiques plus lisses et plus complexes, ce qui améliore la finesse du raisonnement. [SOURCE: Livre p.87, Figure 3-13, p.101]

### 4. Anatomie comparée : Du bloc Original au bloc Moderne
« Observez bien la différence entre la **Figure 3-29** (Original) et la **Figure 3-30** (Moderne, type Llama 3). »

Dans le modèle original (Post-Norm), on faisait le calcul, puis on normalisait. 
🔑 **Dans le modèle moderne (Pre-Norm) :** on normalise **avant** d'entrer dans l'attention ou le FFN. 
⚠️ **Pourquoi ce changement ?** Les chercheurs ont découvert que normaliser avant rend l'entraînement beaucoup plus stable, permettant de monter à des échelles massives sans que le modèle ne "décroche" mathématiquement. [SOURCE: Livre p.124, Figure 3-30]

### 5. Optimisations pour la vitesse et la mémoire (GQA & FlashAttention)
« En tant que futurs ingénieurs, vous devez affronter la réalité : le Transformer est un monstre gourmand en VRAM. » Pour démocratiser l'IA sur des GPU modestes (comme notre T4 sur Colab), des optimisations géniales ont été inventées.

#### Grouped-Query Attention (GQA)
Rappelez-vous la Multi-Head Attention (section 3.1). Avoir 32 têtes pour les Queries, 32 pour les Keys et 32 pour les Values consomme énormément de mémoire.
Comme l'illustrent les **Figures 3-25 à 3-27** (p.120-121) :
*   **Multi-Head Attention (MHA)** : Chaque Query a sa propre Key/Value. (Lourd)
*   **Multi-Query Attention (MQA)** : Toutes les Queries partagent une seule Key/Value. (Trop léger, perd en qualité).
*   **GQA (Le compromis Llama-3)** : On groupe les Queries. Par exemple, 8 têtes de Queries se partagent une seule tête de Key/Value. C'est le meilleur des deux mondes : vitesse du MQA et précision du MHA. [SOURCE: Livre p.121, Figure 3-27]

#### FlashAttention : Le tour de magie matériel
Le goulot d'étranglement n'est pas toujours le calcul, mais le déplacement des données entre la mémoire du GPU (HBM) et son processeur (SRAM).
🔑 **Notez bien cette intuition :** FlashAttention réécrit l'algorithme d'attention pour qu'il tienne entièrement dans la mémoire ultra-rapide du processeur, évitant les allers-retours coûteux. 
Comme décrit p.122, cela permet de doubler la vitesse d'entraînement et de gérer des fenêtres de contexte de 128 000 tokens ou plus ! [SOURCE: Livre p.122, FlashAttention]

### 6. Attention locale et éparse (Sparse Attention)
Regardez la **Figure 3-22** (p.118). Pour des textes très longs, l'attention complète (chaque mot regarde tous les autres) devient impossible. 
*   **Local Attention** : Le mot ne regarde que ses voisins proches (fenêtre glissante).
*   **Sparse Attention** : Le modèle alterne entre des couches d'attention complète et des couches d'attention limitée (Figure 3-23, p.119). C'est ce qui permet à des modèles comme BigBird ou Longformer de "lire" des livres entiers. [SOURCE: Livre p.118-119, Figures 3-22 et 3-23]

### Note d'Éthique et Environnement par le Prof. Henni
⚠️ **Éthique ancrée** : « Mes chers étudiants, l'optimisation n'est pas qu'un défi de code, c'est un impératif écologique. » 
L'entraînement d'un Transformer massif consomme autant d'énergie qu'une petite ville. 
*   **Le coût de l'inefficacité** : Utiliser un modèle non optimisé (sans GQA ou FlashAttention), c'est gaspiller de la ressource énergétique pour le même résultat sémantique.
*   **Démocratisation** : Sans ces optimisations, l'IA resterait l'apanage des trois ou quatre entreprises les plus riches du monde. Optimiser, c'est permettre à un chercheur, une ONG ou une petite entreprise de faire tourner ses propres modèles localement.

🔑 **C'est votre mission :** En tant qu'ingénieurs SCI2070, vous devez toujours chercher le modèle le plus "frugal" qui répond à votre besoin. L'élégance architecturale se mesure aussi à son empreinte carbone. [SOURCE: Livre p.28, Principes de Responsabilité]

### Synthèse de la section
Nous avons vu que le bloc Transformer est une merveille de régulation thermique (Normalisation), de survie (Residuals) et de stockage de motifs (FFN). Nous avons aussi compris que pour passer à l'échelle, nous avons dû ruser avec les mathématiques (GQA) et le matériel (FlashAttention).

« Vous connaissez maintenant la structure du cerveau artificiel. Mais comment ce cerveau "pense-t-il" concrètement quand on lui pose une question ? C'est ce que nous allons voir dans la dernière section théorique : le **Forward Pass** complet et le secret de la vitesse, le **Cache KV**. »

---
*Fin de la section 3.3 (1380 mots environ)*
## 3.4 Forward pass complet et accélération par cache KV (1100+ mots)

### Le voyage de l’information : De la question à la réponse
« Bonjour à toutes et à tous ! Nous arrivons au sommet de notre troisième semaine. Nous avons disséqué les organes du Transformer : ses yeux (l’attention), sa boussole (RoPE) et son squelette (les blocs). Maintenant, il est temps de donner vie à cet ensemble. Nous allons suivre le **Forward Pass**, c'est-à-dire le voyage d'une fraction de seconde que parcourt l'information depuis le moment où vous appuyez sur "Entrée" jusqu'à ce que le premier mot de la réponse apparaisse sur votre écran. 🔑 **Je dois insister :** comprendre ce flux est ce qui distingue un simple utilisateur d'un véritable ingénieur en IA. » [SOURCE: Livre p.74]

### 1. La cascade du Forward Pass (Figures 3-4 à 3-6)
Regardez attentivement la **Figure 3-4 : Les composants du forward pass** (p.98 du livre). Le processus est une cascade linéaire de transformations mathématiques.

1.  **Le Tokenizer (L'entrée)** : Votre phrase est découpée en IDs. Ces entiers sont les adresses de départ.
2.  **La couche d'Embedding** : Les IDs sont transformés en vecteurs denses (vus en 2.4). C'est ici qu'on injecte également l'encodage positionnel (RoPE).
3.  **L'empilement des Blocs (Le cerveau)** : Comme l'illustre la **Figure 3-5** (p.99), le vecteur de chaque token traverse la pile de blocs Transformer (souvent 12, 24 ou même 96 couches). 
    *   🔑 **Note technique** : Dans un décodeur (GPT/Llama), chaque token possède son propre "flux" ou "stream" de calcul. Ils montent les étages de la pile en parallèle, mais ils ne peuvent regarder que vers le bas (les tokens précédents) grâce au masquage d'attention. [SOURCE: Livre p.76-78]
4.  **Le vecteur final** : À la sortie du dernier bloc, nous obtenons un nouveau vecteur pour chaque token d'entrée. Mais pour la génération, seul le vecteur du **dernier token** nous intéresse. Pourquoi ? Parce que c'est lui qui contient la synthèse de tout le contexte nécessaire pour prédire la suite. [SOURCE: Livre p.82, Figure 3-9]

### 2. La Language Modeling Head (LM Head)
Le vecteur qui sort de la pile est un objet mathématique abstrait de dimension 768 ou 4096. Comment le transformer en un mot humain ? 

C'est le rôle de la **LM Head** (Figure 3-6, p.100). 
*   **Projection Linéaire** : On multiplie ce vecteur final par une immense matrice qui le projette dans un espace dont la taille est égale à celle de votre vocabulaire (ex: 50 257 dimensions).
*   **Les Logits** : Nous obtenons des scores bruts appelés **logits**. Un logit élevé pour l'index "42" signifie que le modèle pense très fort que le mot "pomme" est le suivant.
*   **Softmax** : On applique une fonction Softmax pour transformer ces scores en probabilités réelles entre 0 et 1. 

🔑 **La distinction du Professeur Henni** : « Le modèle ne choisit pas un mot. Il calcule une météo de probabilités sur tout le dictionnaire. C'est la stratégie de décodage (Sampling) qui choisira ensuite l'élu parmi les plus probables. » [SOURCE: Livre p.79, p.101]

### 3. Le secret de la vitesse : Le Cache KV (Figure 3-10)
⚠️ **Attention : erreur fréquente ici !** Si vous ne comprenez pas le Cache KV, vous ne comprendrez jamais pourquoi les LLM coûtent si cher à faire tourner.

Le processus de génération est **autorégressif**. Pour générer une phrase de 10 mots, le modèle doit faire 10 forward passes complets. 
*   Passage 1 : Entrée "Le", prédit "chat".
*   Passage 2 : Entrée "Le chat", prédit "mange".
*   Passage 3 : Entrée "Le chat mange", prédit "la"...

Problème : à chaque étape, le modèle doit recalculer l'attention pour les mots qu'il a déjà traités ! C'est un gaspillage monumental. 
🔑 **La solution : Le Cache KV (Key-Value Cache)**. 
Regardez la **Figure 3-10** (p.106). L'idée est de stocker dans la mémoire VRAM du GPU les vecteurs **Key** et **Value** de tous les tokens passés. 
**Analogie** : Imaginez un chef cuisinier qui prépare un repas complexe étape par étape. Au lieu de refaire la sauce à chaque fois qu'il ajoute un nouvel ingrédient dans l'assiette, il garde la sauce prête dans un bol sur le côté. Le Cache KV, c'est ce bol. Le modèle n'a plus qu'à calculer la **Query** du nouveau mot et à la comparer aux **Keys** et **Values** déjà en mémoire. [SOURCE: Livre p.83-84, Figure 3-10]

**Impact sur la performance** : 
Sans cache KV, le temps de génération augmente de façon quadratique avec la longueur du texte. Avec le cache, il devient linéaire. Comme vous le verrez dans l'exercice de laboratoire, l'activation du cache peut diviser le temps de réponse par 5 ou 10 ! [SOURCE: Livre p.85]

### 4. Analyse de structure : Regarder sous le capot
« Pour finir, je veux que vous appreniez à lire la carte d'identité d'un modèle. » En utilisant la bibliothèque `transformers`, nous pouvons imprimer la structure exacte d'un LLM. 

```python
# Installation : pip install transformers
from transformers import AutoModelForCausalLM

# Utilisons TinyLlama (modèle très léger, parfait pour l'analyse)
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

print(model)
```
⚠️ **Fermeté bienveillante** : En exécutant ce code, vous verrez apparaître des termes comme `LlamaDecoderLayer`, `LlamaAttention`, `LlamaRMSNorm`. 🔑 **C'est non-négociable :** vous devez être capables de reconnaître dans cet affichage informatique les blocs théoriques que nous avons étudiés cette semaine. La `LlamaMLP` est votre réseau Feedforward, et le `lm_head` final est votre traducteur vecteur-vers-mots. [SOURCE: Livre p.78, p.100]

### 5. Note d'Éthique par le Prof. Henni : Le coût de la mémoire
⚠️ **Éthique ancrée** : « Mes chers étudiants, le Cache KV est une bénédiction pour la vitesse, mais c'est un fardeau pour les ressources. » 
Le cache KV consomme une quantité massive de mémoire vive sur le GPU (VRAM). 
*   **Inégalité matérielle** : Un modèle avec une fenêtre de contexte de 128 000 tokens nécessite des dizaines de gigaoctets de cache KV. Cela signifie que l'IA de pointe devient inaccessible pour ceux qui n'ont pas de serveurs surpuissants.
*   **Consommation électrique** : Maintenir ces données en cache et multiplier les accès mémoire a un coût énergétique. 

🔑 **C'est votre défi :** En tant qu'experts, vous devrez arbitrer entre la vitesse de réponse (meilleure expérience utilisateur) et la consommation de mémoire. Des techniques comme la **quantification du cache KV** (réduire la précision des vecteurs stockés) sont les nouvelles frontières d'une IA plus sobre et plus accessible. [SOURCE: Livre p.28, Principes de Responsabilité]

### Synthèse de la Semaine 3
« Quel voyage passionnant ! »
1.  Nous avons appris à calculer les **scores d'attention** (Q, K, V).
2.  Nous avons donné un sens de l'ordre au modèle via les **rotations positionnelles (RoPE)**.
3.  Nous avons stabilisé les calculs grâce aux **blocs et aux normalisations**.
4.  Nous avons optimisé la production de texte avec le **Cache KV**.

« Vous avez maintenant une compréhension "moteur" complète. Vous ne voyez plus les LLM comme de la magie, mais comme une série de multiplications matricielles extraordinairement bien orchestrées. La semaine prochaine, nous allons enfin sortir du laboratoire pour utiliser ces modèles sur des tâches concrètes : nous étudierons les **Modèles de Représentation (Encoder-only)** comme BERT pour classer et comprendre le monde ! »

---
*Fin de la section 3.4 (1180 mots environ)*
## 🧪 LABORATOIRE SEMAINE 3 (600+ mots)

**Accroche du Professeur Khadidja Henni** : 
« Bonjour à toutes et à tous ! Nous y sommes : après avoir exploré la théorie des Transformers, il est temps d'ouvrir le capot et de regarder le moteur tourner. 🔑 **Je dois insister :** l'architecture que vous allez manipuler aujourd'hui est le socle de TOUT ce que nous ferons jusqu'à la fin du semestre. Ne vous laissez pas intimider par la structure du modèle : voyez-la comme une suite logique d'étapes de calcul. Respirez, nous allons visualiser l'attention et comprendre comment le KV cache nous fait gagner un temps précieux. C'est parti ! »

---

### 🔹 QUIZ MCQ (10 questions)

1. **Combien de matrices de projection principales sont nécessaires pour le calcul de la self-attention d'une seule tête ?**
   a) 1 (W)
   b) 2 (Wq, Wk)
   c) 3 (Wq, Wk, Wv)
   d) 4 (Wq, Wk, Wv, Wo)
   **[Réponse: c]** [Explication: On projette l'entrée sur trois espaces distincts : Query, Key et Value. SOURCE: Livre p.92, Figure 3-18]

2. **Quel composant permet au modèle de "regarder" différentes parties de l'entrée simultanément sous des angles variés ?**
   a) La couche Feedforward
   b) Le KV Cache
   c) La Multi-head attention
   d) Le RMSNorm
   **[Réponse: c]** [Explication: Les têtes multiples permettent de capturer des relations syntaxiques et sémantiques différentes en parallèle. SOURCE: Livre p.91, Figure 3-17]

3. **Quelle optimisation algorithmique réduit drastiquement la consommation mémoire GPU lors du calcul de l'attention ?**
   a) Word2Vec
   b) FlashAttention
   c) L'encodage sinusoïdal
   d) Le Dropout
   **[Réponse: b]** [Explication: FlashAttention optimise les accès mémoire (IO-awareness) entre la mémoire SRAM et HBM du GPU. SOURCE: Livre p.100]

4. **Quelle est la différence fondamentale entre LayerNorm et RMSNorm ?**
   a) RMSNorm n'utilise pas la moyenne, ce qui la rend plus rapide
   b) LayerNorm est plus récente
   c) RMSNorm nécessite plus de calculs
   d) LayerNorm ne s'applique qu'aux RNN
   **[Réponse: a]** [Explication: RMSNorm simplifie la normalisation en se basant uniquement sur la racine carrée de la moyenne des carrés. SOURCE: Livre p.101]

5. **Que stocke précisément le "KV cache" pour accélérer la génération de texte ?**
   a) Les mots déjà générés sous forme de texte
   b) Les vecteurs Keys et Values des tokens passés
   c) Les gradients du modèle
   d) Les scores de probabilité du vocabulaire
   **[Réponse: b]** [Explication: En stockant les K et V, on évite de recalculer l'attention pour tout le passé à chaque nouveau token. SOURCE: Livre p.83, Figure 3-10]

6. **Combien de têtes d'attention trouve-t-on typiquement dans un modèle comme BERT-base ou GPT-2-small ?**
   a) 1
   b) 8
   c) 12
   d) 96
   **[Réponse: c]** [Explication: La configuration standard "base" utilise 12 têtes d'attention par couche. SOURCE: Livre p.18, Figure 1-21]

7. **Quel mécanisme est responsable du traitement simultané (parallèle) des tokens, contrairement aux RNN ?**
   a) La récurrence
   b) La Self-attention
   c) Le masquage
   d) La couche de sortie
   **[Réponse: b]** [Explication: Comme il n'y a pas de boucle temporelle, tous les tokens peuvent interagir en une seule opération matricielle. SOURCE: Livre p.81, Figure 3-8]

8. **Quelle technique divise les matrices Q, K, V en segments plus petits pour augmenter la capacité de modélisation ?**
   a) La quantification
   b) Le mécanisme de têtes (Heads)
   c) Le pooling
   d) Le bit-shifting
   **[Réponse: b]** [Explication: On divise la dimension totale (ex: 768) par le nombre de têtes (ex: 12) pour avoir des projections de taille 64. SOURCE: Livre p.91]

9. **Quel composant final convertit les activations internes du Transformer en probabilités sur tout le vocabulaire ?**
   a) Le bloc d'attention
   b) La couche d'embedding
   c) La Language Modeling Head (LM Head)
   d) La couche résiduelle
   **[Réponse: c]** [Explication: La tête LM est une couche linéaire suivie d'un softmax projetant vers la taille du dictionnaire. SOURCE: Livre p.76, Figure 3-4]

10. **Quel est l'avantage principal du Rotary Positional Encoding (RoPE) utilisé dans les modèles modernes (Llama, Phi) ?**
    a) Il est plus joli à visualiser
    b) Il capture mieux les relations de position relatives entre les tokens
    c) Il supprime le besoin d'embeddings
    d) Il réduit la taille du vocabulaire
    **[Réponse: b]** [Explication: RoPE applique une rotation complexe aux vecteurs, permettant au modèle de mieux "sentir" la distance entre les mots. SOURCE: Livre p.103-104]

---

### 🔹 EXERCICE 1 : Visualisation de l'attention (Niveau Basique)

**Objectif** : Utiliser un modèle BERT pour extraire les poids d'attention et comprendre comment un token "regarde" ses voisins.

```python
# --- CODE FOURNI (QUESTION) ---
from transformers import AutoModel, AutoTokenizer
import torch

# Chargement du modèle avec l'option de retour d'attention
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True)

sentence = "The cat sat on the mat"
inputs = tokenizer(sentence, return_tensors="pt")

# --- VOTRE TÂCHE : Récupérez les attentions et affichez la forme de la première couche ---

# --- RÉPONSE COMPLÈTE (CORRIGÉ) ---
# [SOURCE: CONCEPT À SOURCER – Documentation Hugging Face & Livre p.89-94]
outputs = model(**inputs)
# 'attentions' est un tuple de 12 tenseurs (un par couche)
attentions = outputs.attentions 

# Récupération de la première couche
first_layer_attention = attentions[0]

print(f"Forme de l'attention (Couche 1) : {first_layer_attention.shape}")
# Attendu : [1, 12, 8, 8] -> [Batch, Heads, Seq_len, Seq_len]
print("Succès : Le modèle a bien généré une matrice d'interaction pour les 12 têtes !")
```

---

### 🔹 EXERCICE 2 : Analyse de structure interne (Niveau Intermédiaire)

**Objectif** : Apprendre à lire l'architecture d'un modèle pour identifier le nombre de couches et la dimension cachée.

```python
# --- CODE FOURNI (QUESTION) ---
from transformers import AutoModelForCausalLM

# Utilisons un modèle léger pour l'analyse
model = AutoModelForCausalLM.from_pretrained("gpt2")

# --- VOTRE TÂCHE : Identifiez le nombre de couches et la dimension d'entrée (n_embd) ---

# --- RÉPONSE COMPLÈTE (CORRIGÉ) ---
# [SOURCE: CONCEPT À SOURCER – Livre p.78 & Structure de classe PyTorch]
print(model) # Affiche la structure complète

# Extraction via la configuration
n_layers = model.config.n_layer
embedding_dim = model.config.n_embd

print(f"\n--- RAPPORT D'ARCHITECTURE ---")
print(f"Nombre de blocs Transformer : {n_layers}")
print(f"Dimension des vecteurs (Model Dim) : {embedding_dim}")
# [SOURCE: Livre p.100 pour les comparaisons de tailles]
```

---

### 🔹 EXERCICE 3 : Mesure de l'impact du KV Cache (Niveau Avancé)

**Objectif** : Démontrer empiriquement l'accélération apportée par le caching des Keys et Values lors de la génération.

```python
# --- CODE FOURNI (QUESTION) ---
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
input_ids = tokenizer("Once upon a time in a galaxy far, far away", return_tensors="pt").input_ids

# --- VOTRE TÂCHE : Comparez le temps de génération de 20 tokens avec et sans cache ---

# --- RÉPONSE COMPLÈTE (CORRIGÉ) ---
# [SOURCE: CONCEPT À SOURCER – Livre p.83-85, Figure 3-10]

# 1. Sans KV Cache
start = time.time()
output_no_cache = model.generate(input_ids, max_new_tokens=20, use_cache=False)
end_no_cache = time.time() - start

# 2. Avec KV Cache
start = time.time()
output_cache = model.generate(input_ids, max_new_tokens=20, use_cache=True)
end_cache = time.time() - start

print(f"Temps SANS cache : {end_no_cache:.4f}s")
print(f"Temps AVEC cache : {end_cache:.4f}s")
print(f"Facteur d'accélération : {end_no_cache/end_cache:.2f}x")

# ⚠️ Note du Professeur : Sur de très longues séquences, l'écart devient massif !
```

---

**Mots-clés de la semaine** : Self-Attention, Multi-head, Query/Key/Value, RoPE, KV Cache, FlashAttention, RMSNorm, Feedforward, LM Head.

**En prévision de la semaine suivante** : Nous allons utiliser ces connaissances pour explorer les modèles spécialisés dans la compréhension : la famille BERT (Encoder-only) et leurs applications en classification. [SOURCE: Detailed-plan.md]

**SOURCES COMPLÈTES** :
*   Livre : Alammar & Grootendorst (2024), *Hands-On LLMs*, Chapitre 3, p.73-106.
*   Blog Jay Alammar : *The Illustrated Transformer* (https://jalammar.github.io/illustrated-transformer/).
*   Kexing : *Transformer Improvements* (https://kexing.info/2023/12/29/transformer-improvements/).
*   GitHub Officiel : chapter03 repository.

[/CONTENU SEMAINE 3]
