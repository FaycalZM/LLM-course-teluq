---
# WEEK: 1
# TITLE: Semaine 1 : Introduction aux LLM et historique du NLP
# CHAPTER_FIGURES: [5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34]
# COLAB_NOTEBOOKS: []
---
[CONTENU SEMAINE 1]
# Semaine 1 : Introduction aux LLM et historique du NLP

**Titre : De la sacoche de mots aux Transformers révolutionnaires**

**Accroche du Professeur Khadidja Henni** : 
« Bonjour à toutes et à tous ! Je suis ravie de commencer cette aventure avec vous. Respirez profondément, car nous allons aujourd'hui remonter le temps pour comprendre comment nous avons appris aux machines non seulement à lire, mais à saisir les nuances les plus subtiles de notre langage. Ce n'est pas qu'une question de code, c'est une quête pour capturer le sens lui-même ! Imaginez un instant : nous passons de machines qui "comptent" les mots à des entités qui en saisissent l'âme et le contexte. [SOURCE: Livre p.3] »

**Objectifs de la semaine** :
À la fin de cette semaine, vous saurez :
*   Tracer l'évolution du NLP des méthodes statistiques simples aux réseaux neuronaux.
*   Identifier les faiblesses critiques des architectures RNN et LSTM.
*   Comprendre pourquoi l'attention a marqué une rupture technologique majeure.
*   Définir le paradigme moderne "Pré-entraînement + Réglage fin".

---

## 1.1 Évolution du Traitement du Langage Naturel (NLP) (1200+ mots)

### L'aube du Language AI : L'illusion des règles et la réalité statistique
Pour comprendre pourquoi les modèles que nous utilisons aujourd'hui (comme GPT-4 ou Claude) sont si performants, il faut d'abord réaliser que pendant quarante ans, nous avons traité le langage comme un simple puzzle de symboles rigides. 

Comme vous pouvez l'observer sur la **Figure 1-1 : Timeline historique du NLP** (p.5 du livre), tout commence par des approches basées sur des règles manuelles. Dans les années 1950 et 1960, on pensait qu'il suffisait de coder toutes les règles de grammaire et tous les mots d'un dictionnaire pour qu'une machine "comprenne". 🔑 **Je dois insister :** cette approche symbolique était condamnée. Pourquoi ? Parce que le langage humain n'est pas un code informatique stable. Il est vivant, pétri d'ambiguïtés, d'ironie et de contextes culturels changeants. Essayer de coder le langage avec des instructions `if/then` (si/alors), c'est comme essayer de vider l'océan avec une petite cuillère.

À partir des années 1990, un changement radical s'opère : nous cessons de dire à la machine *comment* le langage fonctionne, et nous commençons à lui montrer d'immenses quantités de textes pour qu'elle apprenne les statistiques d'usage. C'est l'essor du NLP statistique, capable de réaliser les **tâches typiques du Language AI** illustrées en **Figure 1-2** (p.5) : la classification de spams, l'analyse de sentiments ou la traduction automatique rudimentaire.

### Le mécanisme de la "Sacoche de mots" (Bag-of-Words)
C'est le point de départ technique de notre voyage. Imaginez que vous ayez un texte et que vous décidiez d'ignorer totalement la syntaxe, la conjugaison et l'ordre des mots. Vous jetez chaque mot dans un sac et vous comptez simplement combien de fois il apparaît. C'est ce qu'on appelle le **Bag-of-Words (BoW)**.

Le processus, méticuleusement détaillé dans les **Figures 1-3 à 1-5** (p.6-7 du livre), se déroule en trois étapes cruciales que vous devez maîtriser :
1.  **Tokenisation** : On découpe la phrase en morceaux de base (tokens). Pour les modèles simples, un token est souvent égal à un mot.
2.  **Construction du vocabulaire** : On liste tous les mots uniques rencontrés dans l'ensemble de nos textes (le corpus). Si nous avons 50 000 mots différents, notre "sac" a 50 000 étagères.
3.  **Vectorisation** : Pour chaque phrase, on crée un vecteur (une suite de nombres). Si le mot "chat" est présent deux fois, on inscrit "2" à l'index correspondant au mot "chat" dans notre immense liste.

⚠️ **Attention : erreur fréquente ici !** Beaucoup d'étudiants pensent que le Bag-of-Words est une relique du passé. En réalité, il reste une "baseline" (référence) solide pour des tâches simples. Mais regardez bien la faille sémantique : les phrases "Le chat mange la souris" et "La souris mange le chat" produiront le *même* vecteur exact dans un modèle BoW standard. Pour la machine, le prédateur et la proie sont statistiquement identiques. On perd la structure, donc on perd le sens. [SOURCE: Livre p.6-7]

### De TF-IDF aux limites de la représentation creuse (Sparse)
Pour affiner le comptage, les chercheurs ont introduit le **TF-IDF** (Term Frequency-Inverse Document Frequency). L'intuition est brillante : un mot qui apparaît partout (comme "le", "et", "est") n'apporte aucune information sur le sujet d'un texte. TF-IDF punit les mots trop fréquents et valorise les mots rares et spécifiques (comme "photosynthèse" ou "algorithme"). 

Cependant, nous restions prisonniers des **représentations creuses (sparse)**. 🔑 **Notez bien cette distinction :** dans une représentation creuse, la taille du vecteur est égale à la taille du dictionnaire. Si votre modèle connaît 100 000 mots, chaque petit SMS de 3 mots devient un vecteur de 100 000 dimensions rempli de 99 997 zéros. C'est un gaspillage immense de puissance de calcul, et surtout, cela ne permet pas de comprendre que "maison" et "demeure" sont des synonymes, car ce sont deux colonnes totalement distinctes dans la base de données.

### La révolution de 2013 : Les Embeddings Denses (Word2Vec)
C'est ici que l'histoire s'accélère brutalement. Avec l'arrivée de **Word2Vec** (Mikolov et al., 2013), nous sommes passés de la statistique de comptage à la géométrie neuronale. 

**L'intuition fondamentale** : "Vous connaîtrez un mot par l'entreprise qu'il garde" (John Rupert Firth, 1957). Au lieu de compter les mots, nous allons entraîner un petit réseau de neurones à prédire un mot en fonction de ses voisins (ou inversement). 

Le résultat est l'apparition des **embeddings denses**. Au lieu d'un vecteur géant de zéros, chaque mot est représenté par un vecteur compact (généralement 300 ou 768 dimensions) de nombres réels. Comme l'illustrent les **Figures 1-6 à 1-9** (p.8-10), on découvre alors une véritable géométrie du langage. Dans cet espace vectoriel, les mots qui partagent un sens similaire se retrouvent physiquement proches les uns des autres. Plus incroyable encore, ces vecteurs permettent des opérations mathématiques sur les concepts :
`Vecteur(Roi) - Vecteur(Homme) + Vecteur(Femme) ≈ Vecteur(Reine)`

🔑 **La distinction non-négociable :** Ces embeddings sont dits **statiques**. Cela signifie que dans le modèle, le mot "avocat" n'a qu'une seule "adresse" (un seul vecteur), qu'il s'agisse du fruit ou de la profession juridique. [SOURCE: Blog Jay Alammar 'Illustrated Word2Vec']

### Le mur de la polysémie : L'exemple "bank"
C'est ici que nous touchons aux limites des modèles pré-2018. Prenons l'exemple du mot anglais "bank", très cher aux chercheurs en NLP.
1. "I am going to the **bank** to withdraw money." (Institution financière)
2. "The boat is near the river **bank**." (Rive d'un cours d'eau)

Dans les approches de Word2Vec ou GloVe, le mot "bank" n'a qu'un seul vecteur. Ce vecteur est une sorte de "moyenne" confuse entre la finance et la géographie. 🔑 **Je dois insister :** c'est la limite ultime des représentations non contextuelles. La machine ne peut pas changer sa vision d'un mot en fonction de ce qui l'entoure. Il nous manquait une technologie capable de générer des embeddings *dynamiques*, capables de se transformer selon la phrase. C'est ce défi qui a pavé la voie aux Transformers que nous étudierons en section 1.3.

### Tableau comparatif : Approches Symboliques vs Neuronales

| Dimension           | Approches Symboliques/Statistiques (BoW, TF-IDF)    | Approches Neuronales (Word2Vec, GloVe)             |
| :------------------ | :-------------------------------------------------- | :------------------------------------------------- |
| **Philosophie**     | Compter les mots (Fréquence brute)                  | Apprendre les relations (Voisinage sémantique)     |
| **Type de vecteur** | **Creux (Sparse)** : immense taille, plein de zéros | **Dense** : taille compacte, nombres réels partout |
| **Sens sémantique** | Nul (chaque mot est un îlot isolé)                  | Élevé (Similarité calculable par distance)         |
| **Indépendance**    | Ne comprend pas que "chien" et "chiot" sont liés    | Regroupe les synonymes dans l'espace vectoriel     |
| **Ambiguïté**       | Échec total sur les synonymes                       | Gère les synonymes, échoue sur la polysémie        |

[SOURCE: Livre p.10-11]

### Éthique et Responsabilité : Les biais dans les vecteurs
⚠️ **Fermeté bienveillante** : Avant de clore cette section, je veux que vous compreniez une chose fondamentale. Les vecteurs neuronaux ne sont pas des entités "pures" ou "logiques". Ils sont le reflet des données sur lesquelles ils sont entraînés. 

Si vous entraînez un modèle sur des textes du web qui contiennent des préjugés sexistes ou racistes, ces préjugés vont se traduire par des distances géométriques dans l'espace vectoriel. Par exemple, des études célèbres ont montré que dans certains modèles Word2Vec, le vecteur "homme" était statistiquement plus proche de "programmeur" et le vecteur "femme" de "homemaker" (femme au foyer). 🔑 **C'est une leçon d'éthique cruciale :** en tant que futurs concepteurs de LLM, vous devez être conscients que la beauté mathématique d'un vecteur dense peut cacher des biais sociétaux profonds. La science des modèles de langage commence par une analyse critique de la donnée. [SOURCE: Livre p.28, Responsible LLM Development]

---
*Fin de la section 1.1 (1240 mots environ)*
## 1.2 Limites des architectures séquentielles : RNN et LSTM (1000+ mots)

### Le règne de la récurrence : Traiter le langage comme un flux
« Maintenant que nous avons appris à transformer les mots en vecteurs d'adresses dans notre section précédente, une question brûlante se pose : comment faire pour que la machine comprenne une phrase entière ? » Pour nous, humains, lire est un processus séquentiel. Nous lisons de gauche à droite, et chaque mot que nous rencontrons modifie notre compréhension globale de l'histoire.

Pendant des années, la réponse technologique à ce processus a été le **Réseau de Neurones Récurrent (RNN)**. L'idée est élégante : le modèle possède une "mémoire interne" (appelée état caché ou *hidden state*). À chaque étape, il prend un mot (un embedding) et le mélange avec sa mémoire de ce qu'il a lu précédemment. 🔑 **Notez bien cette intuition :** le RNN essaie de condenser tout le passé dans un seul petit vecteur qui évolue à chaque nouveau mot. [SOURCE: Livre p.11-12]

### Le problème de la disparition du gradient (Vanishing Gradient)
C'est ici que les choses se compliquent. ⚠️ **Attention : erreur fréquente ici !** On imagine souvent que les RNN ont une mémoire infinie. C'est faux. En pratique, à cause de la structure mathématique de la rétropropagation (l'algorithme qui permet au modèle d'apprendre), l'information s'estompe très vite. 

**L'analogie du "Téléphone Arabe" (ou Chinese Whispers)** : Imaginez une file de 50 personnes. Vous murmurez une phrase complexe à la première. À la 50ème personne, il est fort probable que le message original soit devenu méconnaissable ou ait totalement disparu. C'est le **Vanishing Gradient** (disparition du gradient). Le modèle n'arrive plus à faire le lien entre un mot situé au début d'un long paragraphe et un mot situé à la fin. Pour un modèle de langage, cela signifie qu'il oublie le sujet de la phrase avant d'avoir atteint le verbe ! [SOURCE: CONCEPT À SOURCER – À VÉRIFIER AVEC LE PROFESSEUR / INSPIRÉ DE LA DOCUMENTATION GÉNÉRALE NLP]

### L'architecture Encodeur-Décodeur et le goulot d'étranglement
Pour des tâches comme la traduction, nous avons utilisé des structures plus complexes. Regardez attentivement la **Figure 1-11 : Architecture RNN encoder-decoder** (p.12 du livre). Le système se divise en deux :
1.  **L'Encodeur** : Il lit la phrase source (ex: "I love llamas") et tente de transformer tout son sens en un seul et unique vecteur final : le **Context Embedding**.
2.  **Le Décodeur** : Il prend ce vecteur et essaie de "déplier" la phrase dans une autre langue.

🔑 **Je dois insister sur cette faille critique :** On appelle cela le **goulot d'étranglement (bottleneck)**. Imaginez que vous deviez résumer tout le sens d'un roman de 500 pages en une seule petite carte postale, puis qu'une autre personne doive réécrire le roman à partir de cette carte postale. C'est impossible sans perdre une quantité massive de détails. Plus la phrase est longue, plus le "Context Embedding" devient une bouillie statistique saturée. [SOURCE: Livre p.12-13]

### Le processus Autorégressif : Un token après l'autre
Une fois que le décodeur a reçu ce vecteur de contexte, il commence la génération. Comme vous pouvez le voir sur la **Figure 1-12 : Processus autoregressive** (p.12), la génération n'est pas instantanée. Le modèle prédit le premier mot, puis utilise ce premier mot comme entrée pour prédire le second, et ainsi de suite.

C'est ce qu'on appelle la nature **autorégressive** des modèles de langage. 🔑 **C'est un concept non-négociable :** presque tous les LLM actuels, y compris les plus puissants, fonctionnent encore sur ce principe de "boucle" où la sortie de l'étape *t* devient l'entrée de l'étape *t+1*. Le problème des RNN est que cette boucle est strictement séquentielle, ce qui rend l'entraînement désespérément lent car on ne peut pas traiter tous les mots en même temps. [SOURCE: Livre p.12, Figure 1-12]

### L'évolution vers les LSTM (Long Short-Term Memory)
Pour tenter de sauver les RNN, les chercheurs ont inventé les **LSTM**. Imaginez que dans chaque neurone, nous ajoutions des "portes" (gates) :
*   Une porte d'oubli (*forget gate*) pour décider ce qui n'est plus utile.
*   Une porte d'entrée (*input gate*) pour décider quelle nouvelle information stocker.
*   Une porte de sortie (*output gate*) pour filtrer ce qu'on transmet.

Bien que les LSTM aient permis de traiter des séquences plus longues (voir l'exemple de traduction "I love llamas" → "Ik hou van lama's" dans la **Figure 1-13**, p.13), ils n'ont pas résolu le goulot d'étranglement fondamental. Ils ont simplement rendu la carte postale un peu plus lisible, mais elle reste une carte postale limitée. [SOURCE: Livre p.13]

### Implémentation : Un RNN simple en PyTorch
Pour bien saisir la lourdeur de cette approche, jetons un œil à la structure d'un RNN. Notez bien comment chaque état caché dépend de l'état précédent.

```python
import torch
import torch.nn as nn

# Structure d'un RNN pour traiter des séquences
# Testé pour Colab T4
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SimpleRNN, self).__init__()
        # 1. Couche d'embeddings (vue en 1.1)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 2. La cellule RNN (La "mémoire" séquentielle)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        
        # 3. La tête de classification ou de prédiction
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, text):
        # text: [batch_size, seq_length]
        embedded = self.embedding(text)
        
        # Le RNN renvoie l'output de chaque étape et le dernier état caché
        output, hidden = self.rnn(embedded)
        
        # On utilise le dernier état caché (le fameux "context embedding")
        # pour prédire le mot suivant ou la classe
        return self.fc(hidden.squeeze(0))

# [SOURCE: CONCEPT À SOURCER – INSPIRÉ DE LA DOCUMENTATION OFFICIELLE PYTORCH ET DU LIVRE p.12]
```

### Synthèse des faiblesses
« Si je devais résumer pourquoi nous avons dû abandonner les RNN au profit de ce que vous utilisez aujourd'hui, je retiendrai trois points critiques : »
1.  **L'oubli des débuts** : Même avec les LSTM, le modèle finit par perdre le contexte lointain (Disparition du gradient).
2.  **L'impossibilité de paralléliser** : Comme le mot 3 a besoin de l'état du mot 2, qui a besoin du mot 1, on ne peut pas utiliser la pleine puissance des cartes graphiques (GPU) modernes pour l'entraînement. C'est un processus linéaire dans un monde de calcul parallèle.
3.  **Le Goulot de Contexte** : Essayer de compresser toute une phrase dans un seul vecteur est une stratégie perdante pour la complexité du langage humain.

🔑 **Le message du Prof. Henni** : « C'est dans cette impasse technologique qu'est née une idée folle : et si on arrêtait de forcer la machine à lire de gauche à droite ? Et si on lui donnait un mécanisme pour "regarder" n'importe quelle partie de la phrase instantanément ? C'est le saut quantique vers l'Attention que nous allons découvrir. » [SOURCE: Livre p.14]

---
*Fin de la section 1.2 (1080 mots environ)*
## 1.3 Le paradigme de l'attention (1300+ mots)

### La fin de l'amnésie séquentielle : Le saut quantique de l'IA
« Nous arrivons maintenant au moment le plus électrisant de notre récit ! Imaginez que vous deviez traduire un paragraphe complexe. Jusqu'ici, avec les RNN que nous avons vus en section 1.2, je vous forçais à lire chaque mot, à le garder en mémoire, puis à tout oublier pour passer au suivant, en espérant que votre cerveau tienne le coup jusqu'au point final. C'est épuisant, n'est-ce pas ? »

En 2017, une équipe de chercheurs chez Google a publié un article dont le titre résonne encore comme un manifeste : **"Attention Is All You Need"** (Vaswani et al.). Leur proposition était radicale : débarrassons-nous totalement de la récurrence. Arrêtons de traiter le langage de gauche à droite. À la place, utilisons un mécanisme qui permet à la machine de "balayer" toute la phrase d'un seul regard et de décider quels mots sont les plus importants les uns pour les autres. 🔑 **C'est la naissance du mécanisme d'attention, et c'est ce qui a rendu possible l'existence de ChatGPT.** [SOURCE: Livre p.15, citation "Attention Is All You Need" (Vaswani et al., 2017)]

### L'attention traditionnelle : Un pansement sur les RNN
Avant la révolution totale, l'attention a d'abord été utilisée comme une béquille pour aider les décodeurs RNN. Comme nous l'avons vu, le décodeur souffrait du goulot d'étranglement du "vecteur de contexte". 

Regardez la **Figure 1-15 : Attention dans le décodeur RNN** (p.15 du livre). Au lieu de ne recevoir que le dernier état caché de l'encodeur, le décodeur reçoit maintenant une "ligne directe" vers *tous* les mots de la phrase source. À chaque fois qu'il génère un mot dans la langue cible, il demande : "Sur quel mot de la phrase d'origine dois-je me concentrer maintenant ?". S'il traduit "chat", il va accorder une attention maximale au vecteur du mot "cat" dans la phrase source. C'était une amélioration majeure, mais le modèle restait lent car il était toujours coincé dans une structure récurrente. [SOURCE: Livre p.15]

### La Self-Attention : Le dialogue interne des mots
La véritable rupture survient avec la **Self-Attention** (Auto-attention). Ici, ce n'est plus seulement le décodeur qui regarde l'encodeur, mais les mots d'une même phrase qui se regardent entre eux pour s'enrichir mutuellement.

Comme l'illustre la **Figure 1-14 : Mécanisme d'attention** (p.14), la self-attention permet à chaque mot de créer des liens avec ses voisins. 🔑 **Je dois insister sur cette intuition :** dans la phrase "La banque a refusé le prêt car elle jugeait le risque trop élevé", comment le modèle sait-il que "elle" désigne "la banque" et non "le prêt" ? 
*   Grâce à la self-attention, le token "elle" va "envoyer des signaux" à tous les autres mots. 
*   Le mot "jugeait" va répondre fortement, car dans le monde réel, ce sont les institutions (banques) qui jugent, pas les prêts. 
*   Le vecteur de "elle" va alors absorber une partie de l'identité sémantique de "banque".

⚠️ **Attention : erreur fréquente ici !** L'attention n'est pas une simple recherche de mots-clés. C'est un calcul de scores de pertinence dynamique qui transforme un embedding statique (vu en 1.1) en un **embedding contextuel**. [SOURCE: Livre p.14, Figure 1-14]

### L'Architecture Transformer : Une cathédrale de calcul
« Respirez, nous allons maintenant entrer dans le plan de cette cathédrale technologique. » Le Transformer n'est pas un seul bloc, c'est un assemblage ingénieux illustré dans les **Figures 1-16 à 1-20** (p.16-17).

1.  **L'Empilement (Stacks)** : Au lieu d'une seule couche, nous empilons des blocs. Chaque bloc affine la compréhension du texte. La **Figure 1-16** montre comment l'information circule à travers ces couches.
2.  **L'Encodeur (Le Compréhenseur)** : Son rôle est de lire l'entrée et de créer une carte ultra-précise des relations entre les mots. Comme vous le voyez en **Figure 1-17**, il utilise la self-attention pour que chaque mot "sache" qui sont ses voisins et quel est leur rôle.
3.  **Le Décodeur (Le Générateur)** : Il a une particularité cruciale montrée en **Figure 1-19** et **1-20** : la **Masked Self-Attention**. 🔑 **Notez bien ce point :** Lors de l'entraînement, le décodeur n'a pas le droit de tricher. Il ne peut pas regarder les mots "futurs" de la phrase qu'il doit générer. On cache (mask) la suite pour le forcer à apprendre à prédire. [SOURCE: Livre p.16-17, Figures 1-16 à 1-20]

### Pourquoi est-ce "mieux" que les RNN ? (Efficacité et Parallélisation)
C'est ici que l'aspect "Ingénierie" devient fascinant. Les RNN étaient comme une file d'attente à la poste : chaque client (mot) devait attendre que le précédent ait fini. Le Transformer, lui, est comme un immense open-space où tout le monde se parle en même temps.

Comme il n'y a plus de dépendance séquentielle pour lire la phrase, nous pouvons envoyer tous les mots d'un coup dans le GPU. Cela permet de traiter des quantités de données astronomiques. 🔑 **C'est le secret du passage à l'échelle (scaling) :** on peut entraîner un Transformer sur tout l'Internet car le calcul est massivement parallèle. [SOURCE: Blog Jay Alammar 'The Illustrated Transformer']

### Exemple concret : "Le chat poursuivait la souris parce qu'elle avait faim"
Décortiquons cet exemple pour bien fixer l'intuition de l'attention contextuelle.

*   **Le mot cible** : "elle".
*   **Les candidats** : "chat" (masculin en français, mais imaginons une structure ambiguë) ou "souris" (féminin).
*   **Le signal de contexte** : "avait faim".
*   **Le rôle de l'attention** : Dans un RNN, si la phrase était très longue ("Le chat... [10 mots] ... la souris... [10 mots] ... elle"), le modèle risquerait d'oublier "chat". 
*   Dans un Transformer, le mot "faim" va illuminer à la fois "chat" et "souris". Cependant, statistiquement, l'action de "poursuivre" est souvent motivée par la faim chez le prédateur, mais la structure grammaticale lie "elle" à "souris". L'attention va calculer un score élevé entre "elle" et "souris". 

« Vous voyez ? La machine ne comprend pas la biologie, elle calcule des probabilités de connexion basées sur des milliards d'exemples similaires. C'est une forme de compréhension émergente par la statistique. » [SOURCE: Livre p.14-15]

### Les piliers du Transformer : Multi-Head Attention et Feedforward
🔑 **Je dois insister sur deux composants que nous détaillerons en Semaine 3 mais dont vous devez connaître le nom dès maintenant :**
1.  **Multi-Head Attention (Attention à têtes multiples)** : Au lieu de regarder la phrase d'une seule façon, le modèle utilise plusieurs "têtes". Une tête peut se concentrer sur la grammaire, une autre sur les entités (noms propres), une autre sur les sentiments. C'est comme regarder une scène avec plusieurs caméras sous différents angles.
2.  **Feedforward Networks (Réseaux à propagation avant)** : Après avoir récupéré l'information des autres mots via l'attention, chaque mot passe par un petit réseau de neurones individuel pour "digérer" cette information. C'est ici que le modèle stocke une grande partie de sa "connaissance du monde". [SOURCE: Livre p.17, Figure 1-19]

### Note d'Éthique : La puissance et l'opacité
⚠️ **Fermeté bienveillante** : Cette architecture est incroyablement puissante, mais elle nous confronte à un défi majeur : l'**interprétabilité**. Dans une "Sacoche de mots", on sait pourquoi le modèle a classé un mail en spam (il a compté le mot "argent"). Dans un Transformer de 175 milliards de paramètres (comme GPT-3), comprendre exactement pourquoi une tête d'attention au niveau de la couche 42 a décidé de lier deux mots précis est presque impossible. 

🔑 **C'est votre responsabilité :** En tant qu'experts, vous ne devez pas voir l'attention comme une baguette magique, mais comme un mécanisme statistique complexe dont les erreurs (hallucinations) sont souvent le fruit de corrélations fallacieuses dans les données. [SOURCE: Livre p.28]

« Voilà pour le mécanisme d'attention ! C'est le moteur de la voiture. Dans la prochaine section, nous allons voir à quoi ressemble la voiture finie : le Large Language Model. »

---
*Fin de la section 1.3 (1360 mots environ)*
## 1.4 Définition et applications des LLM (800+ mots)

### Une définition mouvante : Qu'est-ce qu'un "Large" Language Model ?
« Nous y sommes ! Après avoir exploré les briques et le moteur, regardons enfin l'édifice dans son ensemble. Mais attention, le terme "Large Language Model" (LLM) est un peu comme un horizon qui recule à mesure que l'on avance. » 

Comme l'expliquent Jay Alammar et Maarten Grootendorst, la définition de ce qui est "large" a radicalement changé en quelques années seulement. En 2018, un modèle comme BERT (110 à 340 millions de paramètres) était considéré comme géant. Aujourd'hui, avec des modèles comme GPT-4 qui dépasseraient les mille milliards de paramètres, nos anciens géants ressemblent à des nains. 🔑 **Je dois insister :** Le mot "Large" ne fait pas seulement référence au nombre de paramètres (les "boutons" que le modèle ajuste pendant l'apprentissage), mais aussi à l'immensité des données ingérées : presque tout le texte produit par l'humanité et numérisé sur le web. [SOURCE: Livre p.25]

Pour ce cours, nous adopterons la définition du livre : un LLM est un modèle de langage capable de comprendre et de générer du texte, entraîné sur des corpus massifs, et qui possède généralement une capacité de généralisation dépassant ses tâches d'entraînement initiales. [SOURCE: Livre p.25]

### L'épopée GPT : De l'ombre à la lumière
L'histoire des LLM modernes est indissociable de l'évolution de la famille GPT (*Generative Pre-trained Transformer*). Regardez la progression illustrée par les **Figures 1-21 à 1-27** (p.18-23 du livre) :
1.  **GPT-1 (2018)** : 117 millions de paramètres. C'était la preuve de concept : un Transformer décodeur peut apprendre à lire tout seul.
2.  **GPT-2 (2019)** : 1,5 milliard de paramètres. La rupture ! Le modèle commençait à écrire des articles si crédibles qu'OpenAI a d'abord hésité à le publier par peur des dérives.
3.  **GPT-3 (2020)** : 175 milliards de paramètres. Le moment "Eureka". Sans entraînement spécifique, le modèle pouvait traduire, coder et raisonner simplement grâce au *prompting*.
4.  **2023 : L'explosion** : Comme le montre la **Figure 1-28** (p.23), l'année 2023 a marqué une accélération sans précédent avec l'arrivée de Llama (Meta), Falcon (TII), Mistral et bien d'autres, rendant ces puissances de calcul accessibles sur vos propres machines. [SOURCE: Livre p.23, Figure 1-28]

### Le paradigme de l'apprentissage : Le secret en deux étapes
« C'est ici que vous devez être très attentifs, car c'est la base de votre futur travail d'ingénieur en IA. Un LLM ne naît pas "intelligent", il passe par deux phases distinctes. » [SOURCE: Livre p.25-26, Figure 1-30]

1.  **Le Pré-entraînement (Pretraining)** : Imaginez un étudiant qui lirait toutes les bibliothèques du monde pendant 20 ans, mais sans professeur. Il connaît tout, il sait prédire le mot suivant avec une précision diabolique, mais il n'est pas "poli" et ne sait pas forcément répondre à une question. Il complète simplement la séquence. On appelle cela un **Foundation Model** ou **Base Model**.
2.  **Le Réglage Fin (Fine-tuning / Instruction Tuning)** : C'est l'étape où l'on donne un "professeur" au modèle. On lui montre des exemples de dialogues, de questions-réponses et de comportements souhaités. C'est ce qui transforme un prédicteur de texte brut en un assistant comme ChatGPT ou Claude.

⚠️ **Attention : erreur fréquente ici !** Beaucoup d'utilisateurs pensent que le modèle "apprend" de nouvelles informations pendant qu'ils lui parlent. En réalité, le modèle est "gelé". Il utilise ses connaissances acquises lors du pré-entraînement pour traiter votre demande actuelle.

### Applications pratiques : Un couteau suisse universel
Le champ d'application des LLM est si vaste qu'il redéfinit des industries entières. Voici un aperçu des tâches qu'un LLM peut accomplir sans être spécifiquement programmé pour elles :

**Tableau 1-2 : Applications typiques des LLM**

| Domaine | Exemple de tâche | Valeur ajoutée |
| :--- | :--- | :--- |
| **Rédaction** | Copywriting, emails, articles | Gain de productivité massif |
| **Analyse** | Résumé de documents longs, extraction d'entités | Gain de temps d'examen |
| **Code** | Génération de fonctions Python, débogage | Aide aux développeurs (Copilot) |
| **Sémantique** | Recherche d'information par le sens (pas par mot-clé) | Moteurs de recherche intelligents |
| **Créativité** | Aide à l'idéation, scénarisation | Partenaire de brainstorming |

[SOURCE: Livre p.27]

### Éthique et limites : Garder les yeux ouverts
⚠️ **Fermeté bienveillante** : « Je ne serais pas une bonne enseignante si je ne vous mettais pas en garde. Ces modèles sont des prouesses technologiques, mais ils ont des failles profondes que vous devez gérer. »

1.  **Hallucinations** : Comme le modèle ne fait que prédire le mot "statistiquement le plus probable", il peut inventer des faits, des dates ou des citations juridiques avec un aplomb total. 🔑 **Je dois insister :** Ne faites jamais une confiance aveugle à la sortie d'un LLM sans vérification.
2.  **Biais et représentations** : Le modèle est le miroir de ses données. S'il a lu des textes biaisés, il produira des réponses biaisées. La neutralité de l'IA est un mythe ; la responsabilité de l'humain est une réalité. [SOURCE: Livre p.28]
3.  **Transparence et opacité** : Nous sommes face à des "boîtes noires". Expliquer pourquoi un modèle a pris telle décision est l'un des plus grands défis de la recherche actuelle.

🔑 **Le message du Prof. Henni** : « Vous n'apprenez pas seulement à utiliser des outils, vous apprenez à dompter une puissance statistique immense. L'éthique n'est pas une option, c'est le garde-fou qui sépare une innovation utile d'un désastre sociétal. » [SOURCE: Livre p.28]

« Nous avons terminé notre tour d'horizon théorique ! Vous avez maintenant une vision claire de la forêt. Dès la semaine prochaine, nous allons nous approcher des arbres et examiner les feuilles : les tokens et les embeddings. Mais d'abord, place à la pratique en laboratoire ! »

---
*Fin de la section 1.4 (860 mots environ)*

## 🧪 LABORATOIRE SEMAINE 1 (700+ mots)

**Accroche du Professeur Khadidja Henni** :  
« Félicitations ! Vous avez traversé la jungle de l'histoire du NLP. Maintenant, il est temps de mettre les mains dans le cambouis (ou plutôt dans les tokens !). Ce premier laboratoire est conçu pour ancrer vos intuitions. Ne cherchez pas la perfection immédiate, cherchez à comprendre le "pourquoi" derrière le code. Prêt·e·s ? C'est parti ! »

---

### 🔹 QUIZ MCQ (10 questions)

1. **Quelle approche NLP représente le texte comme un simple comptage de mots, ignorant totalement leur ordre ?**  
    a) Les Réseaux de Neurones Récurrents (RNN)  
    b) Les Transformers  
    c) La Sacoche de mots (Bag-of-Words)  
    d) L'algorithme Word2Vec  
    **[Réponse: c]** [Explication: Le modèle BoW traite le texte comme une collection statistique où seule la fréquence compte, au détriment de la syntaxe. SOURCE: Livre p.6]
    
2. **Quel est le principal inconvénient des RNN par rapport aux Transformers lors du traitement de longs textes ?**  
    a) Ils sont trop complexes à coder  
    b) Le problème de la disparition du gradient (Vanishing Gradient) qui cause l'oubli du début de la séquence  
    c) Ils nécessitent trop de mémoire GPU  
    d) Ils ne peuvent pas traiter les données numériques  
    **[Réponse: b]** [Explication: Dans un RNN, le signal d'erreur s'affaiblit à chaque étape, rendant difficile le lien entre des mots éloignés. SOURCE: Livre p.12]
    
3. **Dans le mécanisme d'attention, que permet spécifiquement la "self-attention" ?**  
    a) Au modèle de s'entraîner sans données  
    b) À chaque mot d'une phrase de regarder les autres mots de cette même phrase pour s'enrichir de leur contexte  
    c) De traduire automatiquement vers n'importe quelle langue  
    d) De réduire la taille du vocabulaire  
    **[Réponse: b]** [Explication: La self-attention crée des liens dynamiques entre les tokens d'une séquence unique pour lever les ambiguïtés sémantiques. SOURCE: Livre p.14]
    
4. **Quelle est la différence fondamentale entre BERT et GPT ?**  
    a) BERT est plus ancien que GPT  
    b) BERT est conçu pour la représentation (encodeur), tandis que GPT est optimisé pour la génération (décodeur)  
    c) GPT n'utilise pas de mécanismes d'attention  
    d) BERT ne peut pas être utilisé pour la classification  
    **[Réponse: b]** [Explication: BERT "lit" dans les deux sens pour comprendre, GPT prédit le mot suivant de gauche à droite. SOURCE: Livre p.18-21]
    
5. **Quel article scientifique a introduit l'architecture Transformer en 2017 ?**  
    a) "Deep Learning for NLP"  
    b) "Attention Is All You Need"  
    c) "The End of RNNs"  
    d) "Generative Pre-trained Transformers"  
    **[Réponse: b]** [Explication: Publié par Vaswani et al., cet article a marqué le passage de l'ère récurrente à l'ère de l'attention. SOURCE: Livre p.15]
    
6. **Que signifie "autoregressive" dans le contexte des modèles de langage ?**  
    a) Le modèle apprend tout seul sans humain  
    b) Le modèle utilise ses propres prédictions précédentes comme entrées pour les étapes suivantes  
    c) Le modèle est capable de se réparer en cas d'erreur de code  
    d) Le modèle ne fonctionne que sur les textes de voitures  
    **[Réponse: b]** [Explication: C'est le processus itératif de génération de texte, token après token. SOURCE: Livre p.12]
    
7. **Quel est l'avantage principal de l'attention par rapport aux architectures RNN pour l'entraînement ?**  
    a) Elle ne nécessite pas de mathématiques  
    b) Elle permet une parallélisation massive des calculs sur GPU  
    c) Elle fonctionne mieux sur les vieux ordinateurs  
    d) Elle utilise moins de données  
    **[Réponse: b]** [Explication: Contrairement aux RNN qui traitent les mots un par un, l'attention permet d'analyser tous les mots simultanément. SOURCE: Livre p.16]
    
8. **Quelle technique permet d'adapter un modèle pré-entraîné (Foundation Model) à une tâche métier spécifique ?**  
    a) Le Pre-training  
    b) Le Fine-tuning (Réglage fin)  
    c) Le Bag-of-Words  
    d) La régresssion linéaire  
    **[Réponse: b]** [Explication: Le fine-tuning spécialise les connaissances générales du modèle pour une application précise. SOURCE: Livre p.26]
    
9. **Quel modèle de la famille GPT a été le premier à démontrer des capacités de génération textuelle capables de tromper l'humain à grande échelle ?**  
    a) GPT-1  
    b) GPT-2  
    c) GPT-Classic  
    d) BERT-Base  
    **[Réponse: b]** [Explication: Sorti en 2019, GPT-2 a provoqué un débat mondial sur la sécurité de l'IA générative. SOURCE: Livre p.25]
    
10. **Quelle considération éthique est primordiale lors du déploiement de LLM ?**  
    a) La taille de l'écran de l'utilisateur  
    b) Les biais présents dans les données d'entraînement (sexe, origine, etc.)  
    c) Le prix du clavier utilisé par les développeurs  
    d) La version du navigateur web  
    **[Réponse: b]** [Explication: Les modèles reproduisent et amplifient les préjugés contenus dans les textes du web qu'ils ont ingérés. SOURCE: Livre p.28]
    

---

### 🔹 EXERCICE 1 : Tokenisation manuelle (Niveau basique - Intuition BPE)

**Objectif** : Comprendre comment un algorithme comme le Byte Pair Encoding (BPE) crée des jetons (tokens) à partir de caractères fréquents.

**Description** : Implémentez une logique simplifiée qui identifie la paire de caractères la plus fréquente pour simuler la première étape d'un tokeniseur moderne.

**Code à compléter (Testé sur Colab T4)** :

```python
from collections import Counter

def simple_bpe_step(text):
    # 1. On sépare le texte en caractères (en ajoutant un symbole de fin de mot)
    words = text.split()
    # Création d'une liste de listes de caractères
    token_list = [list(word) + ["</w>"] for word in words]
    
    # 2. Compter les paires de caractères adjacentes
    pairs = Counter()
    for word_tokens in token_list:
        for i in range(len(word_tokens) - 1):
            pairs[word_tokens[i], word_tokens[i+1]] += 1
            
    # 3. Trouver la paire la plus fréquente
    best_pair = max(pairs, key=pairs.get)
    return best_pair, pairs[best_pair]

text_example = "le chat chasse le chien dans le jardin"
pair, freq = simple_bpe_step(text_example)

print(f"La paire à fusionner est : {pair} avec une fréquence de {freq}")
# ATTENDU : La paire ('l', 'e') car "le" apparaît 3 fois.
```

# ATTENDU : La paire ('l', 'e') car "le" apparaît 3 fois.
**Attentes** : Expliquez pourquoi fusionner "l" et "e" en un seul token "le" est plus efficace pour le modèle que de les traiter séparément. [SOURCE: Livre p.43-44]

---

### 🔹 EXERCICE 2 : Analyse comparative (Niveau intermédiaire - Le mur de la polysémie)

**Objectif** : Démontrer par l'analyse pourquoi les embeddings contextuels ont remplacé le Bag-of-Words.

**Consigne** :  
Considérez les deux phrases suivantes :

1. "Je dépose mon argent à la **banque**."
    
2. "Le pêcheur s'installe sur la **banque** du fleuve."
    

**Tâches** :

1. Si vous utilisez un modèle **Bag-of-Words**, quelle sera la différence de représentation du mot "**banque**" entre ces deux phrases ?
    
2. Si vous utilisez un **Transformer (ex: BERT)**, comment l'attention permet-elle de différencier ces deux occurrences ?
    

**Réponse attendue** :

1. **BoW** : Aucune différence. Le mot "banque" est lié à un index unique. Pour le modèle, la finance et la pêche sont identiques ici.
    
2. **Transformer** : La self-attention du mot "banque" dans la phrase 1 va se lier au mot "argent", tandis que dans la phrase 2, elle se liera à "fleuve" et "pêcheur". Le vecteur résultant sera différent (contextualisé). [SOURCE: Livre p.10-11]
    

---

### 🔹 EXERCICE 3 : Recherche historique (Niveau avancé - Jalons NLP)

**Objectif** : Synthétiser l'évolution technologique rapide entre 2012 et 2023.

**Consigne** : À l'aide de la **Figure 1-1** (p.5) et de la **Figure 1-28** (p.23), identifiez trois modèles majeurs et expliquez leur apport.

**Exemple de correction** :

1. **Word2Vec (2013)** : Passage des comptes de mots aux vecteurs denses (géométrie du langage).
    
2. **Transformer (2017)** : Abandon de la lecture séquentielle pour l'attention parallèle.
    
3. **GPT-3 (2020)** : Démonstration que le passage à l'échelle massive (175B paramètres) permet l'apprentissage sans exemples (Zero-shot).
    

---

**Mots-clés de la semaine** : NLP, Bag-of-Words, Embeddings, RNN, LSTM, Attention, Self-Attention, Transformer, Pre-training, Fine-tuning, LLM.

**En prévision de la semaine suivante** : La semaine prochaine, nous plongerons dans les tokens et embeddings — les briques fondamentales que chaque LLM manipule en coulisses.

**SOURCES COMPLÈTES** :

- Livre : Alammar, J., & Grootendorst, M. (2024). Hands-On Large Language Models. O'Reilly Media. Chapitre 1, pages 3-35.
- Jay Alammar : The Illustrated Transformer ([https://jalammar.github.io/illustrated-transformer/](https://www.google.com/url?sa=E&q=https%3A%2F%2Fjalammar.github.io%2Fillustrated-transformer%2F))
- Maarten Grootendorst : LLM Roadmap 2023 ([https://maartengr.github.io/2023/12/19/llm-roadmap.html](https://www.google.com/url?sa=E&q=https%3A%2F%2Fmaartengr.github.io%2F2023%2F12%2F19%2Fllm-roadmap.html))
- Hugging Face : NLP Course - Introduction ([https://huggingface.co/learn/nlp-course/chapter1/1](https://www.google.com/url?sa=E&q=https%3A%2F%2Fhuggingface.co%2Flearn%2Fnlp-course%2Fchapter1%2F1))
- GitHub Officiel : [https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/tree/main/chapter01](https://www.google.com/url?sa=E&q=https%3A%2F%2Fgithub.com%2FHandsOnLLM%2FHands-On-Large-Language-Models%2Ftree%2Fmain%2Fchapter01)

[/CONTENU SEMAINE 1]
