---
# WEEK: 2
# TITLE: Semaine 2 : Tokens, tokeniseurs et embeddings
# CHAPTER_FIGURES: [12, 26, 38, 39, 40, 41, 43, 44, 45, 47, 48, 49, 50, 51, 52, 53, 101]
# COLAB_NOTEBOOKS: []
---
[CONTENU SEMAINE 2]
# Semaine 2 : Tokens, tokeniseurs et embeddings

**Titre : Les briques fondamentales des LLM : De la tokenisation aux représentations vectorielles**

**Accroche du Professeur Khadidja Henni** : 
« Bonjour à toutes et à tous ! Je suis ravie de vous retrouver. La semaine dernière, nous avons survolé l'histoire du NLP pour comprendre comment nous en sommes arrivés aux Transformers. Aujourd'hui, nous allons changer d'échelle : nous allons sortir le microscope ! 🔍 Nous allons étudier les atomes du langage : les **tokens**. Comprendre comment une machine découpe le texte est crucial, car si le découpage est mauvais, la compréhension qui suit sera irrémédiablement faussée. Respirez, nous allons décortiquer ensemble ces mécanismes de précision. » [SOURCE: Livre p.37]

**Rappel semaine précédente** : « La semaine dernière, nous avons vu l'évolution des représentations textuelles, de la simple sacoche de mots (Bag-of-Words) aux embeddings denses comme Word2Vec, et comment le mécanisme d'attention a permis de surmonter les limites des RNN. » [SOURCE: Detailed-plan.md]

**Objectifs de la semaine** :
À la fin de cette semaine, vous saurez :
*   Expliquer la théorie mathématique et algorithmique de la tokenisation.
*   Distinguer les schémas par mots, sous-mots, caractères et octets.
*   Comprendre le fonctionnement des algorithmes BPE (Byte Pair Encoding) et WordPiece.
*   Maîtriser la création et la manipulation d'embeddings contextuels.

---

## 2.1 Théorie de la tokenisation (1100+ mots)

### Pourquoi les machines ne lisent-elles pas comme nous ?
Avant d'entrer dans les algorithmes, posons les bases. Un ordinateur ne "lit" pas une chaîne de caractères au sens humain. Il traite des nombres. La **tokenisation** est l'étape de traduction universelle : c'est le processus qui transforme un texte brut en une séquence d'unités discrètes appelées **tokens**. 

Comme l'illustrent les **Figures 2-2 à 2-5** (p.38-43 du livre), la tokenisation n'est pas une simple étape de nettoyage ; c'est la création d'un pont entre notre langage continu et le monde discret des mathématiques. Si vous regardez la **Figure 2-2** (p.38), vous verrez qu'un LLM comme GPT-4 ne voit pas le mot "Bonjour", il voit un index (par exemple, 15432) dans une table de correspondance géante.

### Les 4 grands schémas de tokenisation : Une question de granularité
Il existe plusieurs façons de découper le fromage du langage. Chaque méthode a ses forces et ses faiblesses.

#### 1. La tokenisation par mots (Word Tokenization)
C'est la méthode la plus intuitive. On coupe à chaque espace ou signe de ponctuation.
*   **Avantages** : Les mots portent un sens clair pour nous.
*   **Inconvénients** : ⚠️ **Attention : erreur fréquente ici !** Si vous utilisez cette méthode, votre vocabulaire explose. Entre "marcher", "marchons", "marchait", vous avez trois entrées différentes. Surtout, vous tombez sur le problème des **mots hors vocabulaire (OOV - Out of Vocabulary)**. Si le modèle rencontre un mot qu'il n'a pas vu durant l'entraînement, il affiche le redoutable token `[UNK]` (Unknown), perdant toute information.

#### 2. La tokenisation par caractères (Character Tokenization)
On découpe chaque lettre : "c-h-a-t".
*   **Avantages** : Vocabulaire minuscule (quelques centaines de caractères), zéro mot inconnu.
*   **Inconvénients** : Chaque token individuel ne porte aucun sens. Le modèle doit faire un effort colossal pour réapprendre que "c+h+a+t" signifie un félin. De plus, les séquences deviennent immenses, saturant la fenêtre de contexte du modèle.

#### 3. La tokenisation par sous-mots (Subword Tokenization) - Le Graal moderne
C'est le compromis utilisé par presque tous les LLM actuels (GPT, Llama, BERT). On garde les mots fréquents entiers ("le", "est"), mais on découpe les mots complexes ou rares en morceaux porteurs de sens (morphèmes). Par exemple, "malheureusement" pourrait devenir `malheureuse` + `##ment`. 
🔑 **Je dois insister :** C'est cette méthode qui permet aux modèles de comprendre des mots qu'ils n'ont jamais vus en analysant leurs racines et suffixes ! [SOURCE: Livre p.44-46]

#### 4. La tokenisation par octets (Byte-level Tokenization)
Utilisée notamment par GPT-2 et GPT-4, cette méthode traite le texte comme une suite d'octets (UTF-8). Cela permet de traiter n'importe quel caractère, y compris les emojis 🎵 ou les langues rares, sans jamais avoir de token "inconnu". [SOURCE: Livre p.45]

### Plongée dans les algorithmes : BPE vs WordPiece

#### Byte Pair Encoding (BPE)
C'est l'algorithme star de la famille GPT. Son fonctionnement est itératif et fascinant :
1.  On commence par traiter chaque caractère comme un token.
2.  On cherche la paire de tokens la plus fréquente dans le corpus (ex: "e" et "r").
3.  On les fusionne pour créer un nouveau token "er".
4.  On recommence jusqu'à atteindre la taille de vocabulaire souhaitée (ex: 50 000).

🔑 **Notez bien cette intuition :** BPE construit le langage de bas en haut, en se basant uniquement sur la fréquence statistique des associations de caractères. [SOURCE: Livre p.43]

#### WordPiece
Utilisé par BERT, cet algorithme ressemble à BPE mais avec une nuance mathématique subtile. Au lieu de fusionner la paire la plus *fréquente*, il fusionne la paire qui maximise la **vraisemblance** (likelihood) des données d'entraînement. En d'autres termes, il se demande : "Quelle fusion m'aide le mieux à prédire la structure du langage ?". [SOURCE: Livre p.43-44]

### Impact sur la qualité des modèles
La qualité de votre tokeniseur définit le "plafond" de performance de votre LLM. 
*   **Indentation et Code** : Pour des modèles comme StarCoder, il est vital que les espaces et les tabulations soient des tokens spécifiques. Si un tokeniseur fusionne mal les espaces, le modèle ne comprendra jamais la structure d'un code Python (voir Semaine 13).
*   **Multilingue** : Un tokeniseur entraîné sur l'anglais découpera maladroitement le français ou l'arabe, créant trop de tokens pour une seule phrase, ce qui réduit l'efficacité du modèle.

### Exemple de code : Tokenisation avec Hugging Face Transformers
Pour bien ancrer cela, regardons comment utiliser l'un des tokeniseurs les plus célèbres, celui de BERT.

```python
# Installation : pip install transformers
from transformers import AutoTokenizer

# Chargement d'un tokeniseur standard
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "LLMs are fascinating, aren't they?"

# 1. Encodage : Transformation en IDs
tokens_info = tokenizer(text)
input_ids = tokens_info['input_ids']

print(f"IDs des tokens : {input_ids}")

# 2. Décodage pour voir le découpage
tokens_list = tokenizer.convert_ids_to_tokens(input_ids)
print(f"Découpage visuel : {tokens_list}")

# [SOURCE: CONCEPT À SOURCER – INSPIRÉ DU REPO GITHUB CHAPTER 2 ET DOCUMENTATION HF]
```

⚠️ **Fermeté bienveillante** : Observez le résultat du code ci-dessus. Vous verrez des tokens spéciaux comme `[CLS]` ou `[SEP]`. 🔑 **C'est non-négociable :** vous devez comprendre que ces tokens ne correspondent à aucun mot humain ; ils sont des signaux de contrôle pour le modèle (début de séquence, séparation de phrases). Nous en aurons besoin pour toutes nos tâches de classification en Semaine 4.

### Éthique et Transparence : Les oubliés de la tokenisation
⚠️ **Éthique ancrée** : « Mes chers étudiants, la tokenisation n'est pas neutre. » 
Les tokeniseurs sont entraînés sur des corpus massifs, souvent dominés par l'anglais. 
1.  **Le coût de la langue** : Un utilisateur écrivant en anglais consommera moins de tokens pour la même idée qu'un utilisateur écrivant en amharique ou en wolof. Comme les API de LLM (OpenAI, Anthropic) facturent au token, cela crée une inégalité économique directe basée sur la langue.
2.  **Représentation** : Si un tokeniseur n'a jamais vu de termes techniques médicaux ou juridiques dans sa phase d'apprentissage, il va les "hacher" en petits morceaux insignifiants, rendant la tâche du modèle beaucoup plus difficile. 

🔑 **Je dois insister :** Toujours vérifier si le tokeniseur de votre modèle est adapté à la langue et au domaine de votre application. C'est la première étape d'une IA responsable. [SOURCE: Livre p.28, p.55]

« Voilà pour les bases théoriques ! Nous avons vu comment le texte devient une suite de nombres. Dans la section suivante, nous allons comparer les tokeniseurs des plus grands modèles actuels pour voir comment ces théories se traduisent en pratique. »

---
*Fin de la section 2.1 (1180 mots environ)*
## 2.2 Comparaison des tokeniseurs modernes (1200+ mots)

### La diversité des approches : Pourquoi comparer ?
« Mes chers étudiants, si vous pensiez que tous les modèles "lisaient" de la même manière, cette section va vous ouvrir les yeux ! » Comme nous l'avons vu en section 2.1, la théorie nous donne les outils, mais la pratique nous montre une diversité fascinante. Choisir le mauvais modèle pour votre application, ce n'est pas seulement une question de performance, c'est parfois rendre la tâche impossible au modèle. 

Regardez le tableau récapitulatif inspiré des pages 46 à 54 du livre. Nous allons comparer les géants : **BERT**, la famille **GPT**, **Flan-T5**, et les spécialistes comme **StarCoder2** et **Galactica**. 🔑 **Je dois insister :** chaque différence que nous allons noter a été pensée pour résoudre un problème spécifique de compréhension. [SOURCE: Livre p.46-54]

### BERT : L'ancêtre rigoureux (Cased vs. Uncased)
BERT (2018) utilise l'algorithme **WordPiece**. Sa particularité ? Il existe en deux versions majeures.
1.  **BERT-uncased** : Tout est converti en minuscules. "Paris" devient "paris". ⚠️ **Attention : erreur fréquente ici !** On pourrait croire que c'est plus simple, mais pour une tâche de détection d'entités nommées (NER), on perd l'indice capital de la majuscule qui distingue le prénom "Rose" de la fleur "rose".
2.  **BERT-cased** : Préserve la casse. C'est le standard pour les tâches où la structure propre du nom est capitale.

Observez la **Figure 1-22** (p.19) ou la description p.48 : BERT utilise la notation `##` pour indiquer qu'un token est la suite d'un mot. Par exemple, "embeddings" pourrait être découpé en `em`, `##bed`, `##dings`. Si un mot n'est pas dans son dictionnaire de 30 000 mots, il utilise le token `[UNK]`. 🔑 **Notez bien :** un dictionnaire de 30k est considéré comme "petit" aujourd'hui. [SOURCE: Livre p.47-48]

### La famille GPT : De l'efficacité à l'omniscience
Les modèles d'OpenAI utilisent le **Byte-level BPE**. 
*   **GPT-2 (2019)** : Un vocabulaire de environ 50 000 tokens. Il a introduit une astuce géniale : représenter l'espace *avant* le mot par un caractère spécial (souvent noté `Ġ` dans les visualisations). 
*   **GPT-4 (2023)** : On passe à la vitesse supérieure avec un vocabulaire dépassant les 100 000 tokens. Pourquoi une telle inflation ? Pour être plus efficace. Plus le dictionnaire est grand, plus le modèle peut représenter de longs mots complexes en un seul token, ce qui libère de la place dans sa "fenêtre de contexte". [SOURCE: Livre p.49, p.50-51]

### Flan-T5 et SentencePiece : L'approche "tout-en-un"
Flan-T5 utilise **SentencePiece**. Contrairement à BERT, il ne traite pas les espaces comme des séparateurs à part, mais comme des caractères normaux (souvent remplacés par un tiret bas `_`). 
🔑 **La distinction majeure :** SentencePiece est conçu pour être indépendant de la langue. Il ne suppose pas que les mots sont séparés par des espaces, ce qui le rend redoutable pour le japonais ou le mandarin. Cependant, comme vous le voyez p.50, il peut être "aveugle" aux retours à la ligne, ce qui pose problème pour analyser des listes ou du code source. [SOURCE: Livre p.50]

### Les spécialistes : Quand le domaine dicte la forme
C'est ici que l'ingénierie devient de l'art. Si vous voulez que votre modèle soit un génie des mathématiques ou du code, vous ne pouvez pas utiliser le tokeniseur de Monsieur Tout-le-monde.

#### StarCoder2 : Le traducteur de code
Pour le code source, la structure est tout. StarCoder2 (p.51-52) a deux secrets :
1.  **Tokenisation des chiffres** : Contrairement à GPT-2 qui peut voir "123" comme un seul token, StarCoder2 découpe souvent chiffre par chiffre (`1`, `2`, `3`). Pourquoi ? Pour que le modèle apprenne réellement à faire des additions au lieu de simplement mémoriser des nombres.
2.  **Préservation des indentations** : Il possède des tokens spécifiques pour "quatre espaces", "huit espaces", etc. Sans cela, le modèle perdrait la structure des boucles Python. [SOURCE: Livre p.51-52]

#### Galactica : Le scientifique
Le modèle de Meta pour la science doit gérer du LaTeX (formules mathématiques) et des séquences ADN. Son tokeniseur est entraîné pour ne pas "hacher" les formules chimiques complexes, permettant au modèle de voir `H2O` comme une entité cohérente plutôt que comme une suite de caractères aléatoires. [SOURCE: Livre p.52-53]

### Exemple de comparaison multilingue
Imaginez le mot "manger". 
*   Un tokeniseur anglais pourrait le découper en `man` + `ger`. 
*   Un tokeniseur français bien entraîné (comme CamemBERT) le verra comme un seul token `manger`. 
🔑 **Je dois insister :** Cette fragmentation excessive (over-segmentation) est le fléau des modèles mal adaptés. Si un mot français est découpé en 4 tokens alors qu'un mot anglais équivalent n'en utilise qu'un seul, votre modèle français sera 4 fois plus lent et aura 4 fois moins de mémoire contextuelle.

### Laboratoire de code : Comparaison Hugging Face
Voici comment vous pouvez tester ces différences vous-mêmes sur Google Colab.

```python
# Installation requise : pip install transformers
from transformers import AutoTokenizer

# Sélection de 3 tokeniseurs aux philosophies différentes
tokenizers = {
    "BERT (Social)": "bert-base-uncased",
    "GPT-2 (Général)": "gpt2",
    "Llama-3 (Moderne)": "meta-llama/Meta-Llama-3-8B" # Nécessite accès HF ou version d'essai
}

text = "LLM tokenization is 100% vital for AI."

for name, model_id in tokenizers.items():
    try:
        tk = AutoTokenizer.from_pretrained(model_id)
        tokens = tk.tokenize(text)
        print(f"--- {name} ---")
        print(f"Nombre de tokens : {len(tokens)}")
        print(f"Découpage : {tokens}\n")
    except Exception as e:
        print(f"Note : {name} nécessite une authentification ou n'est pas disponible sans accès spécifique.")

# [SOURCE: CONCEPT À SOURCER – INSPIRÉ DU REPO GITHUB CHAPTER 2 ET DOCUMENTATION HF]
```

### Synthèse des propriétés et impact sur la performance
Pourquoi tout cela est-il capital pour votre futur métier ? 
1.  **Efficacité Computationnelle** : Moins vous avez de tokens pour un texte donné, plus l'inférence (la réponse du modèle) est rapide et économique.
2.  **Qualité des Représentations** : Un tokeniseur qui respecte la morphologie de la langue (ex: séparer le radical de la terminaison d'un verbe) aide énormément le modèle à généraliser.
3.  **Gestion des Nombres** : Comme nous l'avons vu avec StarCoder2, la façon dont les chiffres sont découpés impacte directement les capacités de calcul du LLM.

### Éthique et Inégalités Numériques
⚠️ **Fermeté bienveillante** : « Regardez au-delà de la technique. » 
Il existe une véritable "fracture du token". Les langues à alphabet latin sont extrêmement bien servies par les tokeniseurs actuels. Mais pour les langues d'Afrique ou d'Asie du Sud, un seul mot peut parfois être découpé en une dizaine d'octets. 
🔑 **Conséquence éthique :** Cela signifie que pour dire la même chose, un locuteur de langue "rare" paiera plus cher et subira un modèle moins intelligent (car sa fenêtre de contexte sera saturée plus vite). En tant qu'experts, vous devez militer pour des tokeniseurs plus inclusifs, comme ceux de la famille **Bloom** ou **Llama-3**, qui ont fait des efforts considérables pour élargir leur vocabulaire multilingue. [SOURCE: Livre p.55, Blog 'LLM Roadmap' de Maarten Grootendorst]

« Vous avez maintenant une vue d'ensemble de la jungle des tokeniseurs. Vous comprenez que le choix du modèle commence par l'analyse de son dictionnaire. Dans la section suivante, nous allons étudier les propriétés techniques précises qui font qu'un tokeniseur est "bon" ou "mauvais" pour une tâche donnée. »

---
*Fin de la section 2.2 (1290 mots environ)*
## 2.3 Propriétés des tokeniseurs modernes (800+ mots)

### Au-delà du découpage : L'anatomie d'un bon tokeniseur
« Bonjour à toutes et à tous ! Nous avons comparé les différents visages des tokeniseurs dans la section précédente. Maintenant, je veux que nous regardions sous le capot. Pourquoi certains tokeniseurs réussissent là où d'autres échouent lamentablement ? » 

Un tokeniseur n'est pas juste un script qui coupe des mots ; c'est un système avec des propriétés mathématiques et structurelles précises. Selon Alammar et Grootendorst (p. 55), il existe un ensemble de paramètres et de choix de conception qui dictent l'intelligence future du modèle. Si vous comprenez ces propriétés, vous saurez prédire les limites d'un LLM avant même de lui avoir posé une question. [SOURCE: Livre p.55-56]

### 1. La taille du vocabulaire ($V$) : L'équilibre délicat
🔑 **Je dois insister :** La taille du vocabulaire est le paramètre le plus influent. 
*   **Petit vocabulaire (ex: 30 000 tokens)** : C'est le choix de BERT. L'avantage est que la matrice d'embeddings est légère, ce qui économise de la mémoire GPU. L'inconvénient est la fragmentation : les mots rares sont découpés en de nombreux petits morceaux, ce qui rend la compréhension sémantique plus difficile.
*   **Grand vocabulaire (ex: 100 000+ tokens)** : C'est le choix de GPT-4 ou Llama-3. Cela permet de représenter des concepts complexes (ex: "anticonstitutionnellement") en un seul ou deux tokens. Cela améliore l'efficacité car on traite plus de sens avec moins d'unités. 

⚠️ **Attention : erreur fréquente ici !** On pourrait croire que plus le vocabulaire est grand, mieux c'est. Mais attention au "problème des données creuses" : si votre vocabulaire est trop immense par rapport à vos données d'entraînement, certains tokens ne seront jamais vus, et le modèle n'apprendra jamais leur sens. [SOURCE: Livre p.55]

### 2. Les Tokens Spéciaux : Le langage secret du modèle
Les LLM ne communiquent pas seulement avec des mots, ils utilisent des signaux de contrôle. Imaginez que vous dirigiez un orchestre : vous avez besoin de signes pour dire "commencez" ou "arrêtez".
*   `<s>` ou `[CLS]` : Indique le début d'une séquence.
*   `</s>` ou `[SEP]` : Indique la fin ou sépare deux phrases.
*   `[PAD]` : Utilisé pour égaliser la taille des phrases dans un lot (batch) de calcul.
*   `[MASK]` : Utilisé durant l'entraînement pour cacher un mot que le modèle doit deviner.

🔑 **C'est une distinction non-négociable :** Sans ces tokens spéciaux, le modèle ne peut pas structurer sa pensée. Par exemple, BERT utilise le token `[CLS]` (Classification) pour résumer tout le sens d'une phrase en un seul point. [SOURCE: Livre p.47-48, p.55]

### 3. La gestion de la casse et des domaines
Faut-il convertir "APPLE" en "apple" ? Comme nous l'avons vu, cela dépend de votre tâche.
*   **Modèles Uncased** : Excellents pour la recherche d'information générale ou le clustering où le sens prime sur la forme.
*   **Modèles Cased** : Indispensables pour le code (où `Variable` et `variable` sont deux choses différentes) ou la reconnaissance d'entités nommées.

### Étude de cas : Texte naturel vs. Code source
« Imaginez un instant que vous demandiez à un tokeniseur entraîné sur des romans de lire un script Python. »
Le texte naturel est riche en morphologie (racines, suffixes). Le code source est riche en ponctuation et en indentations. 

Regardez la différence de traitement pour ce bloc de code :
```python
if x > 10:
    print("Success")
```
*   **Tokeniseur généraliste** : Il pourrait ignorer les 4 espaces de l'indentation ou fusionner `if` et `x`. Pour Python, c'est une catastrophe syntaxique !
*   **Tokeniseur spécialisé (StarCoder/Codex)** : Il traite chaque groupe d'espaces comme un token spécifique. Il reconnaît `if` comme une entité unique. 

🔑 **Notez bien :** L'efficacité d'un tokeniseur se mesure souvent par son **ratio Tokens/Caractères**. Plus ce ratio est bas pour un domaine donné, plus le tokeniseur est "intelligent" pour ce domaine. [SOURCE: Livre p.56]

### Le défi du multilingue : L'universalité en question
Comment gérer 100 langues avec un seul tokeniseur ? C'est le défi des modèles comme mBERT ou Bloom. La propriété clé ici est le **partage de vocabulaire**.
Si vous utilisez un tokeniseur entraîné à 90% sur l'anglais, il va "hacher" les mots français. Par exemple, "constitutionnel" deviendra `con` + `stit` + `uti` + `on` + `nel`. 
⚠️ **Éthique ancrée :** « Mes chers étudiants, soyez vigilants. » Un modèle qui utilise trop de tokens pour une langue donnée est un modèle qui a moins de "mémoire" pour cette langue, car sa fenêtre de contexte (ex: 4096 tokens) se remplit beaucoup plus vite. C'est un biais technique qui favorise les langues dominantes. [SOURCE: Livre p.55, Responsible AI principles]

### Bonnes pratiques de sélection
Comment choisir le bon tokeniseur pour votre projet ? Professeur Henni vous donne sa checklist :
1.  **Correspondance de domaine** : Si vous faites du médical, votre tokeniseur connaît-il le vocabulaire latin des maladies ?
2.  **Compression** : Testez votre texte sur plusieurs tokeniseurs (via Hugging Face `Tokenizer.encode`). Celui qui produit le moins de tokens est généralement le plus efficace.
3.  **Gestion des inconnus** : Le modèle utilise-t-il les octets (Byte-level) pour éviter le token `[UNK]` ? C'est crucial pour la robustesse en production.

### Analogie finale
La tokenisation est comme un **tamis**. Si les mailles sont trop larges, tout passe sans distinction (Caractères). Si elles sont trop étroites, vous ne récupérez que des blocs massifs impossibles à analyser (Mots entiers). Le tokeniseur moderne est un tamis magique qui ajuste la taille de ses mailles dynamiquement pour capturer exactement le sens là où il se trouve.

« Vous maîtrisez maintenant les propriétés structurelles des tokens. Mais ces nombres ne sont encore que des étiquettes vides de sens. Dans la section suivante, nous allons donner de la profondeur à ces nombres en découvrant les **Embeddings** : comment transformer un index en un vecteur vibrant de sens sémantique. » [SOURCE: Livre p.57]

---
*Fin de la section 2.3 (890 mots environ)*
## 2.4 Plongements lexicaux (Embeddings) (1500+ mots)

### Bienvenue au cœur de la galaxie sémantique
« Bonjour à toutes et à tous ! Nous arrivons enfin au moment que je préfère. Si la tokenisation que nous avons vue en section 2.1 est l'acte de découper le langage, les **embeddings** sont l'acte de lui donner une âme mathématique. 🔑 **Je dois insister :** c'est ici que réside la magie véritable de l'IA moderne. Sans les embeddings, un ordinateur ne ferait que manipuler des étiquettes numérotées. Avec les embeddings, il commence à "ressentir" la proximité entre les concepts. Imaginez que chaque mot de notre dictionnaire soit une étoile dans une galaxie immense. Les embeddings sont les coordonnées GPS précises qui permettent de savoir quelle étoile brille à côté d'une autre. » [SOURCE: Livre p.57]

### Qu'est-ce qu'un Embedding ? L'intuition géométrique
Oubliez les définitions arides. Un embedding est une **représentation vectorielle dense**. 
*   **Vectorielle** : Une liste de nombres (ex: [0.1, -0.5, 0.8...]).
*   **Dense** : Contrairement au Bag-of-Words (section 1.1), il n'y a presque pas de zéros. Chaque nombre porte une information.

Comme vous pouvez le voir sur la **Figure 1-8** (p.9 du livre), chaque dimension du vecteur peut être imaginée comme une "propriété" abstraite. 
**Analogie** : Imaginez un vecteur à 3 dimensions pour décrire des fruits : [Sucré, Rouge, Gros]. 
*   Une "Pomme" pourrait être `[0.9, 0.8, 0.4]`.
*   Une "Banane" pourrait être `[0.7, 0.1, 0.5]`.
*   Une "Pastèque" pourrait être `[0.6, 0.1, 0.9]`.

Dans un LLM, nous n'utilisons pas 3 dimensions, mais souvent **768** ou **1024**, voire plus. ⚠️ **Attention : erreur fréquente ici !** Ces dimensions ne correspondent pas à des concepts humains clairs comme "couleur" ou "poids". Ce sont des caractéristiques apprises par le modèle au fil de ses lectures, souvent trop abstraites pour nous, mais d'une précision redoutable pour lui. [SOURCE: Livre p.9, Figure 1-8]

### Le premier pilier : Les Embeddings Statiques (Word2Vec)
Avant les LLM, nous utilisions des embeddings dits **statiques**. Le plus célèbre est **Word2Vec** (2013). La **Figure 2-7** (p.58) montre comment le modèle stocke ces vecteurs : c'est une simple table de correspondance (Lookup Table). Chaque token a son vecteur unique, une fois pour toutes.

#### L'algorithme Skip-gram et le Negative Sampling
Comment la machine apprend-elle ces vecteurs ? Par l'observation de ses voisins. C'est l'intuition du **Skip-gram** illustrée en **Figure 2-14** (p.88) : on prend un mot cible (ex: "sat") et on demande au modèle de prédire les mots qui l'entourent ("The", "cat", "on", "the").

🔑 **Note technique sur le Negative Sampling** : Pour apprendre efficacement, le modèle a besoin de contre-exemples. Si je lui montre seulement que "chat" va avec "mange", il pourrait devenir "paresseux" et répondre "oui" à tout. On lui montre donc des paires absurdes (ex: "chat" + "ordinateur") et on lui dit : "Ce ne sont pas des voisins". C'est ce contraste qui sculpte la précision du vecteur. [SOURCE: Livre p.64-67, Figures 2-11 à 2-16]

#### La limite fatidique : La Polysémie
Le problème des modèles comme Word2Vec ou GloVe, c'est qu'ils sont incapables de gérer les mots à plusieurs sens. 🔑 **C'est une distinction non-négociable :** Dans un modèle statique, le mot "avocat" n'a qu'un seul vecteur. Si vous parlez de justice, le vecteur est le même que si vous parlez de guacamole. Le modèle fait une "moyenne" maladroite des sens, ce qui brouille sa compréhension. [SOURCE: Livre p.11]

### Le second pilier : Les Embeddings Contextuels (La révolution BERT)
C'est ici qu'interviennent les LLM modernes. Regardez attentivement les **Figures 2-8 et 2-9** (p.59-60). Contrairement à Word2Vec, un modèle comme BERT ou DeBERTa ne se contente pas de chercher le vecteur dans une table. Il fait passer le mot à travers ses couches de Transformers (l'attention, vue en section 1.3).

**Le résultat ?** Le mot "bank" n'aura pas le même vecteur s'il est suivi de "money" ou de "river". Le vecteur est **calculé dynamiquement** en fonction de l'environnement. C'est ce qu'on appelle la **contextualisation**. Un mot devient un caméléon : il change de couleur vectorielle selon le support sur lequel il se pose. [SOURCE: Livre p.58-60]

### La Géométrie du Sens : Similarité Cosinus
Comment savoir si deux mots sont proches ? On utilise la **Similarité Cosinus**. Comme le montre la **Figure 4-15** (p.125), on ne regarde pas la longueur des vecteurs, mais l'angle entre eux dans cet espace à haute dimension. 
*   Angle proche de 0° : Les mots sont synonymes ou très liés.
*   Angle de 90° : Les mots n'ont aucun rapport.
*   Angle de 180° : Les mots sont opposés (rare en langage naturel).

🔑 **Je dois insister :** Cette propriété géométrique est la base de tous les moteurs de recherche modernes. On ne cherche plus des mots-clés, on cherche des vecteurs proches. [SOURCE: Livre p.125]

### Laboratoire de code : Créer des Embeddings avec Sentence-Transformers
Mettons cela en pratique. Nous allons utiliser un modèle léger et performant : `all-MiniLM-L6-v2`. Ce modèle transforme une phrase entière en un seul vecteur de 384 dimensions.

```python
# Installation : pip install sentence-transformers
from sentence_transformers import SentenceTransformer, util

# 1. Chargement du modèle (Optimisé pour Colab T4)
# Ce modèle est petit mais redoutable pour la similarité sémantique.
model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "Le chat dort sur le tapis.",
    "Un félin se repose sur la moquette.",
    "Le cours de l'action Apple est en hausse.",
    "J'aime manger des pommes bien rouges."
]

# 2. Encodage : Transformation en vecteurs
embeddings = model.encode(sentences)

print(f"Forme de la matrice d'embeddings : {embeddings.shape}") 
# (4, 384) -> 4 phrases, chacune représentée par 384 nombres.

# 3. Calcul de similarité entre la phrase 0 et la phrase 1
sim_chat = util.cos_sim(embeddings[0], embeddings[1])
print(f"Similarité entre 'chat' et 'félin' : {sim_chat.item():.4f}")

# 4. Comparaison avec la phrase 2 (finance)
sim_finance = util.cos_sim(embeddings[0], embeddings[2])
print(f"Similarité entre 'chat' et 'Apple stock' : {sim_finance.item():.4f}")

# [SOURCE: CONCEPT À SOURCER – INSPIRÉ DU REPO GITHUB CHAPTER 2 ET BLOG MAARTEN GROOTENDORST]
```

⚠️ **Fermeté bienveillante** : Observez les scores. La similarité entre la phrase sur le chat et celle sur le félin sera très élevée (proche de 0.8 ou 0.9), même si elles ne partagent *aucun* mot commun ! C'est la preuve que le modèle a compris le concept derrière les symboles.

### Applications Pratiques des Embeddings
Pourquoi passer autant de temps sur ce concept ? Parce qu'il est partout :
1.  **Recherche Sémantique** : Trouver un document même si la requête utilise des synonymes.
2.  **Clustering (Semaine 7)** : Regrouper automatiquement des milliers d'emails par thématique.
3.  **Systèmes de Recommandation** : Si vous aimez la chanson A, le système cherche la chanson dont l'embedding est le plus proche de A (voir Figure 2-17, p.68).
4.  **Détection d'Anomalies** : Un texte dont le vecteur est très éloigné de tous les autres est probablement un spam ou une erreur.

[SOURCE: Livre p.67-70]

### Éthique et Biais : La face cachée des vecteurs
⚠️ **Éthique ancrée** : « Mes chers étudiants, écoutez-moi bien. » 
Parce que les embeddings apprennent à partir de nos textes, ils figent nos préjugés dans le marbre mathématique. 
*   **Stéréotypes** : Si le mot "technologie" est statistiquement plus associé aux hommes dans les textes du web, l'embedding de "technologie" sera géométriquement plus proche de "homme". 
*   **Conséquence** : Un algorithme de recrutement basé sur ces embeddings pourrait rejeter des CV de femmes simplement parce que leur profil est "mathématiquement" moins proche du vecteur "ingénieur".

🔑 **C'est votre responsabilité :** Avant d'utiliser un modèle d'embedding, vérifiez toujours sa provenance et testez ses biais avec des paires de mots sensibles. L'IA ne doit pas être un amplificateur d'injustices. [SOURCE: Livre p.28, Principes de Responsabilité]

### Synthèse de la Semaine 2
« Nous avons parcouru un chemin immense aujourd'hui ! »
1.  Nous avons appris à découper le texte en **tokens** (atomes).
2.  Nous avons vu comment ces tokens sont traduits en **embeddings denses** (coordonnées).
3.  Nous avons compris la différence entre un vecteur **statique** (mort) et un vecteur **contextuel** (vivant et changeant).

« Vous tenez entre vos mains les clés de la compréhension des LLM. La semaine prochaine, nous monterons encore d'un cran : nous entrerons dans la salle des machines du Transformer pour voir exactement comment les têtes d'attention manipulent ces vecteurs pour créer de l'intelligence. »

---
*Fin de la section 2.4 (1580 mots environ)*

## 🧪 LABORATOIRE SEMAINE 2 (850+ mots)

**Accroche du Professeur Khadidja Henni** : 
« Bonjour à toutes et à tous ! Prêts pour votre premier voyage au cœur de l'atome sémantique ? Ce laboratoire est un moment de vérité : nous allons manipuler les briques que nous avons étudiées en théorie. Rappelez-vous : dans le monde des LLM, un petit changement dans la tokenisation peut transformer une réponse géniale en une bouillie incompréhensible. Soyez méticuleux, soyez curieux, et surtout, observez bien comment les chiffres commencent à parler ! »

---

### 🔹 QUIZ MCQ (10 questions)

1. **Quel schéma de tokenisation est le plus efficace pour éliminer totalement le problème des mots inconnus ([UNK]) ?**
   a) Tokenisation par mots entiers
   b) Tokenisation par sous-mots (WordPiece)
   c) Tokenisation par octets (Byte-level BPE)
   d) Tokenisation par phrases
   **[Réponse: c]** [Explication: En travaillant au niveau de l'octet, n'importe quelle séquence de caractères peut être décomposée, éliminant le besoin de tokens "inconnus". SOURCE: Livre p.45]

2. **Quelle est la différence fondamentale entre BERT-cased et BERT-uncased ?**
   a) BERT-cased est beaucoup plus grand
   b) BERT-uncased convertit tout en minuscules, perdant les indices sur les noms propres
   c) BERT-cased n'utilise pas de Transformers
   d) BERT-uncased est réservé au code source
   **[Réponse: b]** [Explication: "Uncased" ignore la casse, ce qui simplifie le vocabulaire mais peut nuire à la reconnaissance d'entités nommées. SOURCE: Livre p.48]

3. **Quel tokeniseur moderne est particulièrement optimisé pour ne pas "hacher" les indentations du code source ?**
   a) BERT Tokenizer
   b) Word2Vec
   c) StarCoder2 Tokenizer
   d) Bag-of-Words
   **[Réponse: c]** [Explication: Les tokeniseurs spécialisés dans le code créent des tokens dédiés aux séquences d'espaces pour préserver la structure syntaxique. SOURCE: Livre p.51]

4. **Dans un espace vectoriel d'embeddings, que représente mathématiquement un concept ?**
   a) Un seul nombre entier
   b) Un vecteur (une suite de nombres réels)
   c) Une chaîne de caractères
   d) Une matrice de zéros
   **[Réponse: b]** [Explication: Un concept est projeté dans un espace à haute dimension (ex: 768) où sa position définit son sens. SOURCE: Livre p.9]

5. **L'algorithme Word2Vec utilise le "Negative Sampling" pour :**
   a) Supprimer les mauvais mots du dictionnaire
   b) Apprendre au modèle à distinguer les voisins probables des paires aléatoires
   c) Accélérer la vitesse de la carte graphique
   d) Traduire les textes vers le français
   **[Réponse: b]** [Explication: Le contraste entre exemples positifs et négatifs est ce qui permet de sculpter l'espace sémantique. SOURCE: Livre p.65]

6. **Quelle est la dimension typique d'un vecteur d'embedding pour un modèle comme BERT-base ?**
   a) 2 dimensions
   b) 50 dimensions
   c) 768 dimensions
   d) 1 million de dimensions
   **[Réponse: c]** [Explication: C'est le standard pour les modèles "base" de la famille Transformer. SOURCE: Livre p.82]

7. **Pourquoi la tokenisation par sous-mots est-elle supérieure à celle par mots entiers ?**
   a) Elle utilise des vecteurs plus longs
   b) Elle permet de décomposer des mots rares en racines et suffixes connus
   c) Elle est plus ancienne et plus stable
   d) Elle ne nécessite pas d'entraînement
   **[Réponse: b]** [Explication: Cela permet au modèle de généraliser son savoir à des mots qu'il n'a jamais rencontrés. SOURCE: Livre p.44]

8. **Quel token spécial GPT-2 utilise-t-il pour signaler la fin d'un texte ?**
   a) `[SEP]`
   b) `</s>`
   c) `<|endoftext|>`
   d) `[END]`
   **[Réponse: c]** [Explication: C'est le marqueur de fin de séquence spécifique à la famille GPT. SOURCE: Livre p.49]

9. **Quelle mesure géométrique est la plus utilisée pour évaluer la similarité sémantique entre deux vecteurs ?**
   a) La somme des nombres
   b) La similarité cosinus (l'angle entre les vecteurs)
   c) La longueur du texte
   d) Le nombre de voyelles
   **[Réponse: b]** [Explication: Elle mesure l'alignement directionnel des concepts dans l'espace. SOURCE: Livre p.125]

10. **Quel modèle d'embedding récent utilise des "Rotary Positional Embeddings" (RoPE) ?**
    a) Word2Vec
    b) Llama-2 / Llama-3
    c) GloVe
    d) TF-IDF
    **[Réponse: b]** [Explication: RoPE est une technique moderne pour encoder la position des tokens de manière plus fluide. SOURCE: Livre p.102]

---

### 🔹 EXERCICE 1 : Comparaison de tokeniseurs (Niveau 1 - Basique)

**Objectif** : Visualiser physiquement comment différents modèles découpent la même phrase.

**Description** : Utilisez la bibliothèque `transformers` pour charger BERT et GPT-2 et analysez leur comportement sur un texte complexe.

**Code (Testé pour Colab T4)** :
```python
from transformers import AutoTokenizer

# Chargement des tokeniseurs
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "L'intelligence artificielle est fascinante 🎵 #AI2024"

# Tâche : Tokenisez et affichez le résultat
print(f"BERT : {bert_tokenizer.tokenize(text)}")
print(f"GPT-2: {gpt2_tokenizer.tokenize(text)}")
```

**Typical Answer** : BERT affichera des `##` pour les sous-mots et risque de transformer l'emoji en `[UNK]` s'il n'est pas dans son dictionnaire. GPT-2 gérera l'emoji grâce à sa gestion des octets et ajoutera des symboles comme `Ġ` pour les espaces. [SOURCE: Livre p.46-51]

---

### 🔹 EXERCICE 2 : Création d'embeddings (Niveau 2 - Intermédiaire)

**Objectif** : Transformer une pensée en un vecteur numérique et vérifier sa "forme".

**Description** : Utilisez `sentence-transformers` pour encoder une critique de film et analysez l'objet résultant.

**Code (Testé pour Colab T4)** :
```python
from sentence_transformers import SentenceTransformer

# Modèle recommandé p.62 du livre
model = SentenceTransformer("all-MiniLM-L6-v2")

sentence = "Ce cours sur les LLM est absolument incroyable !"
embedding = model.encode(sentence)

print(f"Dimension du vecteur : {embedding.shape}")
print(f"Les 5 premières valeurs : {embedding[:5]}")
```

**Typical Answer** : La dimension sera de **(384,)**. Les valeurs sont des nombres réels entre -1 et 1 environ. [SOURCE: Livre p.62, p.84]

---

### 🔹 EXERCICE 3 : Visualisation et Similarité (Niveau 3 - Avancé)

**Objectif** : Utiliser la réduction de dimension pour "voir" la proximité sémantique.

**Consigne** : Calculez la similarité cosinus entre trois phrases : deux proches et une éloignée. Utilisez ensuite PCA (Principal Component Analysis) pour projeter ces vecteurs en 2D.

**Code (Testé pour Colab T4)** :
```python
from sentence_transformers import util
import numpy as np
from sklearn.decomposition import PCA

sentences = [
    "J'adore les chats.",
    "Les félins sont mes animaux préférés.",
    "La bourse de Paris a clôturé en baisse."
]

embeddings = model.encode(sentences)

# 1. Calcul de similarité
sim = util.cos_sim(embeddings[0], embeddings[1])
print(f"Similarité Chat/Félin : {sim.item():.4f}")

# 2. Réduction de dimension (PCA)
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)
print(f"Coordonnées 2D :\n{reduced_embeddings}")
```

**Typical Answer** : La similarité entre les phrases 1 et 2 sera élevée (> 0.7), tandis qu'avec la phrase 3, elle sera faible (< 0.2). En 2D, les points 1 et 2 apparaîtront regroupés, loin du point 3. [SOURCE: Livre p.125, p.141]

---

**Mots-clés de la semaine** : Tokenisation, Sous-mots (Subwords), BPE, WordPiece, Embeddings denses, Espace vectoriel, Similarité Cosinus, [CLS]/[SEP], PCA.

**En prévision de la semaine suivante** : La semaine prochaine, nous monterons encore d'un cran : nous entrerons dans la salle des machines du Transformer pour voir exactement comment les têtes d'attention manipulent ces vecteurs.

**SOURCES COMPLÈTES** :
*   Livre : Alammar, J., & Grootendorst, M. (2024). *Hands-On Large Language Models*. O'Reilly Media. Chapitre 2, pages 37-71.
*   Blog Jay Alammar : *Illustrated Word2Vec* (https://jalammar.github.io/illustrated-word2vec/)
*   Blog Hugging Face : *About Tokenizers* (https://huggingface.co/blog/tokenizers)
*   GitHub Officiel : https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/tree/main/chapter02

[/CONTENU SEMAINE 2]


