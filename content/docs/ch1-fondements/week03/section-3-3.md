---
title: "3.3 Blocs Transformer et optimisation"
weight: 4
---

{{< katex />}}

## L'architecture du sanctuaire : Le bloc Transformer

Bonjour à toutes et à tous ! Nous avons vu comment les mots se parlent à travers l'attention (3.1) et comment ils se repèrent dans l'espace via RoPE (3.2). Mais imaginez maintenant que vous essayiez de faire tenir cette conversation dans un ouragan permanent. Sans une structure pour stabiliser les signaux, l'information se perdrait dans un chaos mathématique total.

{{% hint info %}}
🔑 **Je dois insister :** ce que nous appelons "Le Transformer", ce n'est pas juste l'attention, c'est l'assemblage de ce que nous appelons les **Blocs Transformer**. Un modèle comme GPT-4 en empile des dizaines. C'est dans cette répétition, dans cette stratification du savoir, que naît l'intelligence émergente.
{{% /hint %}}

## 1. Les connexions résiduelles : L'autoroute de l'information

Regardez la **Figure 3-13 : Bloc Transformer original**. Vous remarquerez des flèches qui "sautent" par-dessus les couches d'attention et de réseau de neurones. C'est ce qu'on appelle les **Residual Connections** (ou Skip Connections).

{{< bookfig src="82.png" week="03" >}}

**L'intuition du Professeur Henni** : Imaginez que vous deviez transmettre un message à travers 100 intermédiaires. À chaque étape, le message risque d'être déformé. Une connexion résiduelle, c'est comme si vous donniez à chaque intermédiaire une copie scellée du message original en lui disant : "Ajoute tes remarques sur un post-it, mais ne touche pas à l'original".

{{% hint info %}}
🔑 **Pourquoi est-ce vital ?** 
Sans ces connexions, nous souffririons du problème de la disparition du gradient (vu en 1.2). Les connexions résiduelles créent une "autoroute" directe qui permet au signal de l'erreur de redescendre jusqu'aux premières couches sans être étouffé par les calculs complexes de l'attention. Mathématiquement, on écrit : $Sortie = x + SousCouche(x)$. C'est le fameux "Add" dans le diagramme "Add & Norm".
{{% /hint %}}

## 2. La normalisation : Le régulateur de tension

{{% hint warning %}}
**Attention : erreur fréquente ici !** On pense souvent que plus les nombres sont grands dans un réseau de neurones, plus le modèle est "puissant". C'est l'inverse ! Des valeurs qui explosent rendent le modèle instable et impossible à entraîner. Il nous faut un mécanisme de stabilisation : la **Normalisation**.
{{% /hint %}}

*   **LayerNorm (L'approche classique)** : Introduite dans le Transformer original, elle calcule la moyenne et la variance de toutes les activations d'une couche pour les ramener à une échelle standard (centrée sur 0 avec un écart-type de 1).
*   **RMSNorm (L'approche moderne - Llama/Phi)** : Comme vous le voyez en **Figure 3-14**, les modèles récents utilisent le **Root Mean Square Layer Normalization**. 

{{< bookfig src="83.png" week="03" >}}

{{% hint info %}}
🔑 **Je dois insister sur cette distinction d'ingénierie :** RMSNorm est beaucoup plus rapide car elle ne calcule pas la moyenne, seulement la racine carrée de la moyenne des carrés. C'est une simplification qui ne perd rien en performance mais qui fait gagner des millisecondes précieuses sur des milliards de calculs.
{{% /hint %}}

## 3. Le réseau Feedforward (FFN) : La banque de connaissances

Après l'attention, chaque mot passe par un **Feedforward Neural Network** (souvent appelé MLP pour Multi-Layer Perceptron). Si l'attention sert à "récupérer" l'information des voisins, le FFN sert à "traiter" et à "stocker" cette information.

Regardez la **Figure 3-15**. Le FFN est composé de deux couches linéaires avec une fonction d'activation au milieu.

{{< bookfig src="66.png" week="03" >}}

*   **Intuition** : L'attention dit "Le mot 'banque' ici parle d'argent". Le FFN, lui, fouille dans sa mémoire interne pour activer toutes les associations liées à la finance.
*   **Évolution technique** : On est passé de la fonction ReLU à la fonction **SwiGLU**. SwiGLU permet au modèle d'apprendre des fonctions mathématiques plus lisses et plus complexes, ce qui améliore la finesse du raisonnement.

## 4. Anatomie comparée : Du bloc Original au bloc Moderne

Observez bien la différence entre la **Figure 3-13** (Original) et la **Figure 3-14** (Moderne, type Llama 3).

Dans le modèle original (Post-Norm), on faisait le calcul, puis on normalisait. 

{{% hint info %}}
🔑 **Dans le modèle moderne (Pre-Norm) :** on normalise **avant** d'entrer dans l'attention ou le FFN.
{{% /hint %}}

{{% hint warning %}}
**Pourquoi ce changement ?** Les chercheurs ont découvert que normaliser avant rend l'entraînement beaucoup plus stable, permettant de monter à des échelles massives sans que le modèle ne "décroche" mathématiquement.
{{% /hint %}}

## 5. Optimisations pour la vitesse et la mémoire (GQA & FlashAttention)

En tant que futurs ingénieurs, vous devez affronter la réalité : le Transformer est un monstre gourmand en VRAM. Pour démocratiser l'IA sur des GPU modestes (comme notre T4 sur Colab), des optimisations géniales ont été inventées.

### Grouped-Query Attention (GQA)

Rappelez-vous la Multi-Head Attention (section 3.1). Avoir 32 têtes pour les Queries, 32 pour les Keys et 32 pour les Values consomme énormément de mémoire.
Comme illustré dans les **Figures 3-9, 3-8 et 3-16** :
*   **Multi-Head Attention (MHA)** : Chaque Query a sa propre Key/Value. (Lourd)
*   **Multi-Query Attention (MQA)** : Toutes les Queries partagent une seule Key/Value. (Trop léger, perd en qualité).
*   **GQA (Le compromis Llama-3)** : On groupe les Queries. Par exemple, 8 têtes de Queries se partagent une seule tête de Key/Value. C'est le meilleur des deux mondes : vitesse du MQA et précision du MHA.

{{< bookfig src="80.png" week="03" >}}

### FlashAttention : Le tour de magie matériel

Le goulot d'étranglement n'est pas toujours le calcul, mais le déplacement des données entre la mémoire du GPU (HBM) et son processeur (SRAM).

{{% hint info %}}
🔑 **Notez bien cette intuition :** FlashAttention réécrit l'algorithme d'attention pour qu'il tienne entièrement dans la mémoire ultra-rapide du processeur, évitant les allers-retours coûteux. 
Cela permet de doubler la vitesse d'entraînement et de gérer des fenêtres de contexte de 128 000 tokens ou plus !
{{% /hint %}}

## 6. Attention locale et éparse (Sparse Attention)

Regardez la **Figure 3-17**. Pour des textes très longs, l'attention complète (chaque mot regarde tous les autres) devient impossible. 

{{< bookfig src="75.png" week="03" >}}

*   **Local Attention** : Le mot ne regarde que ses voisins proches (fenêtre glissante).
*   **Sparse Attention** : Le modèle alterne entre des couches d'attention complète et des couches d'attention limitée (Figure 3-18). C'est ce qui permet à des modèles comme BigBird ou Longformer de "lire" des livres entiers.

{{< bookfig src="76.png" week="03" >}}

## Éthique et Environnement

{{% hint danger %}}
« Mes chers étudiants, l'optimisation n'est pas qu'un défi de code, c'est un impératif écologique. »

L'entraînement d'un Transformer massif consomme autant d'énergie qu'une petite ville. 
*   **Le coût de l'inefficacité** : Utiliser un modèle non optimisé (sans GQA ou FlashAttention), c'est gaspiller de la ressource énergétique pour le même résultat sémantique.
*   **Démocratisation** : Sans ces optimisations, l'IA resterait l'apanage des trois ou quatre entreprises les plus riches du monde. Optimiser, c'est permettre à un chercheur, une ONG ou une petite entreprise de faire tourner ses propres modèles localement.

🔑 **C'est votre mission :** En tant qu'ingénieurs, vous devez toujours chercher le modèle le plus "frugal" qui répond à votre besoin. L'élégance architecturale se mesure aussi à son empreinte carbone.
{{% /hint %}}

## Synthèse de la section

Nous avons vu que le bloc Transformer est une merveille de régulation thermique (Normalisation), de survie (Residuals) et de stockage de motifs (FFN). Nous avons aussi compris que pour passer à l'échelle, nous avons dû ruser avec les mathématiques (GQA) et le matériel (FlashAttention).

Vous connaissez maintenant la structure du cerveau artificiel. Mais comment ce cerveau "pense-t-il" concrètement quand on lui pose une question ? C'est ce que nous allons voir dans la dernière section théorique : le **Forward Pass** complet et le secret de la vitesse, le **Cache KV**.
