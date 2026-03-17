---
title: "1.3 Le paradigme de l'attention"
weight: 4
---

## La fin de l'amnésie séquentielle : Le saut quantique de l'IA

Nous arrivons maintenant au moment le plus électrisant de notre récit ! Imaginez que vous deviez traduire un paragraphe complexe. Jusqu'ici, avec les RNN que nous avons vus en section 1.2, je vous forçais à lire chaque mot, à le garder en mémoire, puis à tout oublier pour passer au suivant, en espérant que votre cerveau tienne le coup jusqu'au point final. C'est épuisant, n'est-ce pas ?

En 2017, une équipe de chercheurs chez Google a publié un article dont le titre résonne encore comme un manifeste : **"Attention Is All You Need"** (Vaswani et al.). Leur proposition était radicale : débarrassons-nous totalement de la récurrence. Arrêtons de traiter le langage de gauche à droite. À la place, utilisons un mécanisme qui permet à la machine de "balayer" toute la phrase d'un seul regard et de décider quels mots sont les plus importants les uns pour les autres. 🔑 **C'est la naissance du mécanisme d'attention, et c'est ce qui a rendu possible l'existence de ChatGPT.**

## L'attention traditionnelle : Un pansement sur les RNN

Avant la révolution totale, l'attention a d'abord été utilisée comme une béquille pour aider les décodeurs RNN. Comme nous l'avons vu, le décodeur souffrait du goulot d'étranglement du "vecteur de contexte".

Regardez la **Figure 1-14 : Attention dans le décodeur RNN**. Au lieu de ne recevoir que le dernier état caché de l'encodeur, le décodeur reçoit maintenant une "ligne directe" vers *tous* les mots de la phrase source. À chaque fois qu'il génère un mot dans la langue cible, il demande : "Sur quel mot de la phrase d'origine dois-je me concentrer maintenant ?". S'il traduit "chat", il va accorder une attention maximale au vecteur du mot "cat" dans la phrase source. C'était une amélioration majeure, mais le modèle restait lent car il était toujours coincé dans une structure récurrente.

{{< bookfig src="19.png" week="01" >}}

## La Self-Attention : Le dialogue interne des mots

La véritable rupture survient avec la **Self-Attention** (Auto-attention). Ici, ce n'est plus seulement le décodeur qui regarde l'encodeur, mais les mots d'une même phrase qui se regardent entre eux pour s'enrichir mutuellement.

Comme l'illustre la **Figure 1-13 : Mécanisme d'attention**, la self-attention permet à chaque mot de créer des liens avec ses voisins. 🔑 **Je dois insister sur cette intuition :** dans la phrase "La banque a refusé le prêt car elle jugeait le risque trop élevé", comment le modèle sait-il que "elle" désigne "la banque" et non "le prêt" ?
*   Grâce à la self-attention, le token "elle" va "envoyer des signaux" à tous les autres mots.
*   Le mot "jugeait" va répondre fortement, car dans le monde réel, ce sont les institutions (banques) qui jugent, pas les prêts.
*   Le vecteur de "elle" va alors absorber une partie de l'identité sémantique de "banque".

{{< bookfig src="18.png" week="01" >}}

{{% hint warning %}}
**Attention : erreur fréquente ici !** L'attention n'est pas une simple recherche de mots-clés. C'est un calcul de scores de pertinence dynamique qui transforme un embedding statique (vu en 1.1) en un **embedding contextuel**.
{{% /hint %}}

## L'Architecture Transformer : Une cathédrale de calcul

Respirez, nous allons maintenant entrer dans le plan de cette cathédrale technologique. Le Transformer n'est pas un seul bloc, c'est un assemblage ingénieux illustré dans les **Figures 1-15 à 1-19**.

1.  **L'Empilement (Stacks)** : Au lieu d'une seule couche, nous empilons des blocs. Chaque bloc affine la compréhension du texte. La **Figure 1-15** montre comment l'information circule à travers ces couches.
2.  **L'Encodeur (Le Compréhenseur)** : Son rôle est de lire l'entrée et de créer une carte ultra-précise des relations entre les mots. Comme vous le voyez en **Figure 1-16**, il utilise la self-attention pour que chaque mot "sache" qui sont ses voisins et quel est leur rôle.
3.  **Le Décodeur (Le Générateur)** : Il a une particularité cruciale montrée en **Figure 1-18** et **1-19** : la **Masked Self-Attention**. 🔑 **Notez bien ce point :** Lors de l'entraînement, le décodeur n'a pas le droit de tricher. Il ne peut pas regarder les mots "futurs" de la phrase qu'il doit générer. On cache (mask) la suite pour le forcer à apprendre à prédire.

{{< bookfig src="20.png" week="01" >}}

{{< bookfig src="21.png" week="01" >}}

{{< bookfig src="22.png" week="01" >}}

{{< bookfig src="23.png" week="01" >}}

{{< bookfig src="24.png" week="01" >}}

## Pourquoi est-ce "mieux" que les RNN ? (Efficacité et Parallélisation)

C'est ici que l'aspect "Ingénierie" devient fascinant. Les RNN étaient comme une file d'attente à la poste : chaque client (mot) devait attendre que le précédent ait fini. Le Transformer, lui, est comme un immense open-space où tout le monde se parle en même temps.

Comme il n'y a plus de dépendance séquentielle pour lire la phrase, nous pouvons envoyer tous les mots d'un coup dans le GPU. Cela permet de traiter des quantités de données astronomiques. 🔑 **C'est le secret du passage à l'échelle (scaling) :** on peut entraîner un Transformer sur tout l'Internet car le calcul est massivement parallèle.

## Exemple concret : "Le chat poursuivait la souris parce qu'elle avait faim"

Décortiquons cet exemple pour bien fixer l'intuition de l'attention contextuelle.

*   **Le mot cible** : "elle".
*   **Les candidats** : "chat" (masculin en français, mais imaginons une structure ambiguë) ou "souris" (féminin).
*   **Le signal de contexte** : "avait faim".
*   **Le rôle de l'attention** : Dans un RNN, si la phrase était très longue ("Le chat... [10 mots] ... la souris... [10 mots] ... elle"), le modèle risquerait d'oublier "chat".
*   Dans un Transformer, le mot "faim" va illuminer à la fois "chat" et "souris". Cependant, statistiquement, l'action de "poursuivre" est souvent motivée par la faim chez le prédateur, mais la structure grammaticale lie "elle" à "souris". L'attention va calculer un score élevé entre "elle" et "souris".

Vous voyez ? La machine ne comprend pas la biologie, elle calcule des probabilités de connexion basées sur des milliards d'exemples similaires. C'est une forme de compréhension émergente par la statistique.

## Les piliers du Transformer : Multi-Head Attention et Feedforward

{{% hint info %}}
🔑 **Je dois insister sur deux composants que nous détaillerons en Semaine 3 mais dont vous devez connaître le nom dès maintenant :**
1.  **Multi-Head Attention (Attention à têtes multiples)** : Au lieu de regarder la phrase d'une seule façon, le modèle utilise plusieurs "têtes". Une tête peut se concentrer sur la grammaire, une autre sur les entités (noms propres), une autre sur les sentiments. C'est comme regarder une scène avec plusieurs caméras sous différents angles.
2.  **Feedforward Networks (Réseaux à propagation avant)** : Après avoir récupéré l'information des autres mots via l'attention, chaque mot passe par un petit réseau de neurones individuel pour "digérer" cette information. C'est ici que le modèle stocke une grande partie de sa "connaissance du monde".
{{% /hint %}}

## Note d'Éthique : La puissance et l'opacité

{{% hint danger %}}
Cette architecture est incroyablement puissante, mais elle nous confronte à un défi majeur : l'**interprétabilité**. Dans une "Sacoche de mots", on sait pourquoi le modèle a classé un mail en spam (il a compté le mot "argent"). Dans un Transformer de 175 milliards de paramètres (comme GPT-3), comprendre exactement pourquoi une tête d'attention au niveau de la couche 42 a décidé de lier deux mots précis est presque impossible.

🔑 **C'est votre responsabilité :** En tant qu'experts, vous ne devez pas voir l'attention comme une baguette magique, mais comme un mécanisme statistique complexe dont les erreurs (hallucinations) sont souvent le fruit de corrélations fallacieuses dans les données.
{{% /hint %}}

Voilà pour le mécanisme d'attention ! C'est le moteur de la voiture. Dans la prochaine section, nous allons voir à quoi ressemble la voiture finie : le Large Language Model.
