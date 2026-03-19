---
title: "3.2 Encodage positionnel"
weight: 3
---

{{< katex />}}

## Le paradoxe du Transformer : Une intelligence sans boussole

Imaginez, mes chers étudiants, que je vous donne une boîte remplie de mots découpés. Je les jette sur une table et je vous demande : "Quelle est l'histoire ?". Vous seriez bien en peine de me répondre ! C'est précisément le problème du Transformer que nous avons vu en section 3.1. 

Contrairement aux RNN (section 1.2) qui lisent naturellement de gauche à droite, le mécanisme d'attention est **invariant par permutation**. 

{{% hint info %}}
🔑 **Je dois insister sur ce point technique :** pour le calcul de self-attention, la phrase "Le chat mange la souris" et "La souris mange le chat" sont strictement identiques si l'on ne regarde que les vecteurs. Pour le modèle, c'est juste un ensemble de points dans l'espace qui se parlent. Sans un mécanisme supplémentaire, le Transformer est incapable de faire la différence entre le prédateur et la proie. Il nous faut donc injecter une "boussole" temporelle : l'**encodage positionnel**.
{{% /hint %}}

## L'approche historique : Les ondes sinusoïdales

Dans l'article original de 2017, les chercheurs ont eu une idée poétique : utiliser des fonctions trigonométriques (sinus et cosinus) pour marquer la position.
Imaginez que chaque mot porte une étiquette avec un signal sonore unique qui change légèrement selon sa place dans la phrase. Le mot à la position 1 a une fréquence rapide, le mot à la position 100 a une fréquence lente. 

En ajoutant ces valeurs mathématiques aux embeddings denses (vus en section 2.4), le modèle peut déduire la distance entre deux mots. Cependant, cette méthode dite d'**encodage absolu** a une faille majeure : elle a du mal à gérer des phrases plus longues que celles vues pendant l'entraînement. C'est comme si votre boussole s'arrêtait de fonctionner au-delà de 512 mètres.

## La révolution moderne : Rotary Positional Embeddings (RoPE)

Aujourd'hui, presque tous les modèles de pointe (Llama-3, Phi-3, Mistral) utilisent une technique beaucoup plus élégante : les **Rotary Positional Embeddings (RoPE)**. Regardez attentivement les **Figures 3-10 et 3-11**. 

{{< bookfig src="85.png" week="03" >}}
{{< bookfig src="86.png" week="03" >}}

**L'intuition fondamentale** : Au lieu d'ajouter un nombre au vecteur, on le fait **pivoter** dans l'espace. 
**Analogie** : Imaginez deux danseurs (deux tokens) sur une piste. Pour savoir s'ils sont proches l'un de l'autre dans la phrase, on ne regarde pas seulement leur position sur la piste, mais aussi l'angle de leur corps. S'ils ont pivoté de la même façon, ils sont proches. S'ils ont un décalage d'angle important, ils sont éloignés.

{{% hint info %}}
🔑 **Pourquoi est-ce supérieur ?** 
1.  **Relation relative** : Le modèle se moque de savoir si un mot est à la position 500 ou 501. Ce qui compte, c'est que la distance entre eux est de 1. RoPE capture magnifiquement cette information relative.
2.  **Extrapolation** : Comme il s'agit de rotations (un cycle de 360°), le modèle peut théoriquement traiter des séquences beaucoup plus longues que prévu (ce qu'on appelle le *long context window*).
{{% /hint %}}

## L'Ingénierie de l'efficacité : Le Packing

{{% hint warning %}}
**Attention : erreur fréquente ici !** On pense souvent que le modèle traite une phrase, puis s'arrête, puis traite la suivante. En réalité, pour ne pas gaspiller la puissance des GPU, nous utilisons le **Packing** (empaquetage).
{{% /hint %}}

Regardez la **Figure 3-12 : Packing de documents**. Comme beaucoup de documents sont plus courts que la fenêtre de contexte (ex: 4096 tokens), nous les "entassons" les uns après les autres dans une seule séquence, séparés par un token spécial. 

{{< bookfig src="84.png" week="03" >}}

{{% hint info %}}
🔑 **Je dois insister :** l'encodage positionnel doit alors être réinitialisé pour chaque nouveau document à l'intérieur de la même séquence. Sans cela, le modèle croirait que le début du deuxième email est la suite directe de la fin du premier ! C'est une prouesse d'ingénierie logicielle indispensable pour un entraînement rapide et efficace.
{{% /hint %}}

## Visualisation mathématique simplifiée de RoPE

Ne soyez pas effrayés par les mathématiques, l'idée est visuelle. Pour chaque paire de dimensions $(d_i, d_{i+1})$ de notre vecteur, on applique une matrice de rotation :
$$\begin{pmatrix} x_i \\ x_{i+1} \end{pmatrix} \rightarrow \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} x_i \\ x_{i+1} \end{pmatrix}$$
Où $m$ est la position du mot. Chaque mot "tourne" d'un angle proportionnel à sa place dans la phrase. 

C'est magnifique, n'est-ce pas ? Le sens (l'embedding) et l'ordre (la rotation) fusionnent en une seule entité mathématique.

## Éthique : Le biais de position

{{% hint danger %}}
Mes chers étudiants, l'encodage positionnel n'est pas qu'un détail technique. Il influence la façon dont l'IA accorde de l'importance aux informations.

Il existe un phénomène documenté appelé **"Lost in the Middle"** (Perdu au milieu). Les LLM ont tendance à mieux se souvenir des informations situées tout au début ou toute à la fin d'un texte, et à oublier ce qui se trouve au milieu. 

🔑 **C'est une leçon de vigilance :** Lorsque vous concevez un système basé sur des LLM (comme le RAG que nous verrons en Semaine 9), l'ordre dans lequel vous présentez les documents au modèle peut radicalement changer sa réponse. L'IA, comme nous, peut être victime d'un biais de primauté ou de récence.
{{% /hint %}}

## Pourquoi est-ce vital pour vous ?

Comprendre l'encodage positionnel vous permet de :
1.  **Dépanner des modèles** qui perdent le fil sur des textes longs.
2.  **Optimiser l'entraînement** en utilisant intelligemment le packing.
3.  **Choisir le bon modèle** : aujourd'hui, si un modèle n'utilise pas RoPE ou une variante (comme ALiBi), il est souvent considéré comme technologiquement dépassé pour les contextes longs.

Vous voyez maintenant comment les mots se parlent (Attention) et comment ils se repèrent (Position). Mais pour que tout cela fonctionne sans que le cerveau du modèle n'explose, nous avons besoin d'une structure rigide et protectrice : c'est le **Bloc Transformer** et ses mécanismes de normalisation que nous allons disséquer maintenant.
