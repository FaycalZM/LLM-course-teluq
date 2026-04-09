---
title: "3.2 Encodage positionnel"
weight: 3
---

{{< katex />}}

## Le paradoxe du Transformer : La mémoire sans l'ordre
Bonjour à toutes et à tous ! J'espère que vous avez encore en tête notre "soirée bruyante" de la section 3.1. Nous avons vu que le Transformer est un génie de la simultanéité : il peut regarder tous les mots d'un livre en un clin d'œil grâce à la self-attention. Mais, mes chers étudiants, cette puissance a un prix terrifiant. 

> [!IMPORTANT]
📌 **Je dois insister sur ce paradoxe :** de par sa construction mathématique, le Transformer est **invariant par permutation**. 

Cela signifie que pour lui, les phrases "Le chat mange la souris" et "La souris mange le chat" sont rigoureusement identiques. Pourquoi ? Parce que l'attention calcule des scores entre des vecteurs sans se soucier de leur place dans la file d'attente. 

Sans une boussole pour indiquer l'ordre, notre cathédrale de calcul n'est qu'un sac de mots sophistiqué. Aujourd'hui, nous allons apprendre à donner le sens du temps et de l'espace à nos modèles.

---
## L'intuition : Les coordonnées GPS du langage
🧩 Imaginez que vous receviez les pièces d'un puzzle, mais que toutes les pièces soient parfaitement carrées et lisses. Vous savez ce qu'il y a sur chaque pièce (l'embedding sémantique), mais vous n'avez aucune idée de l'endroit où elles s'emboîtent. 

**L'encodage positionnel**, c'est l'étiquette que l'on colle au dos de chaque pièce pour dire : "Je suis la pièce n°1, tout en haut à gauche". 

Dans les RNN (Section 1.2), l'ordre était implicite : le mot 2 arrivait forcément après le mot 1. Dans le Transformer, nous devons injecter cette information artificiellement. 

> [!WARNING]
⚠️ **Attention : erreur fréquente ici !** On ne donne pas simplement un numéro (1, 2, 3...) au modèle. Pourquoi ? Parce que si la phrase est très longue, le nombre "1000" écraserait par sa valeur mathématique les autres informations du vecteur. Nous avons besoin d'une méthode plus subtile.


---
## La méthode classique : Les ondes sinusoïdales
Dans l'article original de 2017, les chercheurs ont utilisé des fonctions sinus et cosinus. 
*   **L'idée** : Chaque position dans la phrase est associée à une fréquence d'onde unique. 
*   **Le bénéfice** : Cela permet au modèle de comprendre la distance relative. Si le modèle sait comment oscille l'onde entre la position 2 et la position 5, il peut généraliser cette "distance de 3" à n'importe quel endroit du texte.

Cependant, cette méthode "absolue" (on ajoute l'information au début du voyage) a montré ses limites lorsque nous avons voulu créer des modèles capables de lire des textes de plus en plus longs. C'est là qu'intervient la révolution de **l'encodage rotatif**.


---
## La révolution RoPE (Rotary Positional Embeddings)
Si vous regardez les spécifications de modèles comme **Llama-3**, **Mistral** ou **Phi-3**, vous verrez toujours mentionné "**RoPE**". C'est aujourd'hui le standard absolu. 

Regardons attentivement la **Figure 3-7 : Application des Rotary Embeddings** .

{{< bookfig src="85.png" week="03" >}}

**ℹ️ Explication** : Cette illustration est fondamentale pour comprendre la différence de philosophie. 
*   **Ancien monde** : On ajoutait la position une seule fois, tout au début, sur les embeddings d'entrée (les boîtes bleues en haut).
*   **Monde RoPE** : Comme le montre la figure, l'encodage positionnel est injecté **à chaque couche**, directement à l'intérieur des blocs d'attention (les ronds violets). 

> [!NOTE]
🔑 **Je dois insister :** RoPE n'est pas une addition, c'est une **multiplication**. On ne "colle" pas une étiquette, on fait "pivoter" le vecteur.


### La mathématique de la rotation
Passons à la géométrie avec la **Figure 3-8 : La rotation des vecteurs** . 

{{< bookfig src="86.png" week="03" >}}

**ℹ️ Explication** : Imaginez que chaque paire de dimensions dans votre vecteur (votre Query ou votre Key) soit une aiguille sur une horloge. 
*   Pour le mot n°1, on tourne l'aiguille de 10 degrés.
*   Pour le mot n°2, on la tourne de 20 degrés.
*   **Le miracle du produit scalaire** : Lorsque le modèle calcule l'attention entre deux mots, la mathématique de la rotation fait que le score final dépend uniquement de **l'angle entre les deux aiguilles**. 
*   Si les mots sont proches, l'angle est petit, le score est fort. S'ils sont loin, l'angle est grand, le score faiblit.


> [!TIP]
💭 **Mon intuition :** RoPE permet au modèle de "sentir" la distance entre les mots sans avoir besoin de connaître leur position absolue. 

> C'est comme si, dans une file d'attente, vous ne saviez pas que vous étiez le 50ème, mais que vous sentiez exactement que la personne devant vous est à 50 cm et celle de derrière à 50 cm. C'est l'**Attention Relative**.


---
## Pourquoi RoPE a-t-il gagné ?
1.  **Extrapolabilité** : Un modèle entraîné sur des phrases de 2048 mots peut, grâce à RoPE, comprendre (un peu mieux) des phrases de 4000 mots car il comprend la logique de rotation.
2.  **Stabilité** : Les rotations préservent la norme (la "longueur") des vecteurs, ce qui évite que le modèle ne devienne instable pendant l'entraînement.
3.  **Richesse sémantique** : En faisant varier la vitesse de rotation selon les dimensions, le modèle peut dévouer certaines parties de son cerveau aux relations à court terme (mots voisins) et d'autres aux relations à long terme (début et fin de paragraphe).


---
## Optimisation de l'entraînement : Le Packing
*Mes chers étudiants, l'informatique n'est pas qu'une affaire de mathématiques, c'est aussi une affaire d'économie.* 

Entraîner un LLM coûte des millions d'euros en électricité. Chaque seconde où votre GPU ne calcule rien est un gaspillage. 

Regardons la **Figure 3-9 : Packing des documents** .

{{< bookfig src="84.png" week="03" >}}

**ℹ️ Explication** : Elle compare deux méthodes d'organisation des données.
*   **Approche naïve (Haut)** : Si vous avez une phrase de 10 mots et une fenêtre de contexte de 2048, vous remplissez le reste avec du "Padding" (des zéros). Le GPU passe son temps à multiplier des zéros. C'est un désastre d'efficacité.
*   **Approche par Packing (Bas)** : On "compacte" plusieurs documents différents à la suite dans le même bloc de 2048 tokens, séparés par un token spécial. 

> [!NOTE]
⚔️ **Le défi technique** : Grâce aux encodages positionnels modernes, le modèle est capable de comprendre que même s'ils sont dans le même bloc, le Document n°2 recommence à la position 1. Sans cela, le modèle croirait que le début du deuxième article est la suite logique de la fin du premier.

---
## Limites et Frontières : La fenêtre de contexte
> [!WARNING]
⚠️ Ne croyez pas que la mémoire de l'IA soit infinie. 

Même avec RoPE, chaque modèle possède une "Context Window" (Fenêtre de contexte) maximale.
*   **La limite physique** : Si le modèle a été entraîné avec une rotation maximale correspondant à 8000 tokens, lui en donner 100 000 va le rendre "étourdi". Les angles de rotation deviennent trop serrés, et il perd le fil de la logique.
*   **Le coût quadratique** : Rappelez-vous la section 3.1. Même si l'encodage positionnel est parfait, le calcul de l'attention demande toujours $N \times N$ opérations. Doubler la fenêtre de contexte multiplie par quatre le besoin en mémoire vive du GPU.

---
## Laboratoire de réflexion : Le temps est-il une dimension ?

> [!CAUTION]
⚠️ Mes chers étudiants, réfléchissez à l'impact de ce découpage.

Pour un Transformer, le temps n'existe pas. Il n'y a que des positions dans une grille. 
1.  **L'absence de causalité réelle** : Le modèle ne comprend pas que la cause précède l'effet parce que c'est une loi physique ; il le comprend parce que statistiquement, le token "Cause" a une position inférieure au token "Effet" dans ses données d'entraînement. 
2.  **Le biais de position** : On a remarqué que les modèles accordent souvent plus d'importance aux informations situées au début et à la fin d'un texte (le phénomène "*Lost in the Middle*"). C'est une conséquence directe de la façon dont nous encodons les positions. 


> [!TIP]
📢 **Mon conseil** : Lorsque vous construisez un système de RAG (Semaine 9), assurez-vous que l'information cruciale ne se trouve pas perdue au milieu d'un énorme bloc de texte, car l'encodage positionnel sémantique y est souvent moins "vif".


---
## Synthèse
Nous avons vu comment le Transformer, initialement aveugle à l'ordre, a acquis une boussole spatio-temporelle. 
*   **L'encodage absolu** (sinus) a posé les bases.
*   **L'encodage rotatif (RoPE)** a apporté la flexibilité et la notion de distance relative, permettant l'explosion des fenêtres de contexte que nous connaissons aujourd'hui.
*   **Le Packing** garantit que nos GPU travaillent à 100% de leur capacité (pas de gaspillage).

> [!TIP]
🔑 **Mon message** : L'ordre des mots est la structure de notre pensée. 

> En apprenant à faire pivoter des vecteurs dans l'espace complexe, les chercheurs ont réussi l'impossible : garder la puissance du calcul parallèle tout en respectant la mélodie séquentielle du langage humain. C'est un triomphe de l'ingénierie mathématique.

---
Vous savez maintenant comment le Transformer regarde et comment il se repère. Mais un cerveau ne se résume pas à ses yeux. Dans la section suivante ➡️, nous allons étudier la "matière grise" du modèle : les **Blocs Transformer** et comment nous les optimisons pour qu'ils ne brûlent pas vos serveurs.