---
title: "3.4 Forward pass complet et accélération par cache KV"
weight: 5
---

## Le voyage de l'information : De la question à la réponse

Bonjour à toutes et à tous ! Nous arrivons au sommet de notre troisième semaine. Nous avons disséqué les organes du Transformer : ses yeux (l'attention), sa boussole (RoPE) et son squelette (les blocs). Maintenant, il est temps de donner vie à cet ensemble. Nous allons suivre le **Forward Pass**, c'est-à-dire le voyage d'une fraction de seconde que parcourt l'information depuis le moment où vous appuyez sur "Entrée" jusqu'à ce que le premier mot de la réponse apparaisse sur votre écran.

{{% hint info %}}
🔑 **Je dois insister :** comprendre ce flux est ce qui distingue un simple utilisateur d'un véritable ingénieur en IA.
{{% /hint %}}

## 1. La cascade du Forward Pass

Regardez attentivement la **Figure 3-19 : Les composants du forward pass**. Le processus est une cascade linéaire de transformations mathématiques.

{{< bookfig src="57.png" week="03" >}}

1.  **Le Tokenizer (L'entrée)** : Votre phrase est découpée en IDs. Ces entiers sont les adresses de départ.
2.  **La couche d'Embedding** : Les IDs sont transformés en vecteurs denses (vus en 2.4). C'est ici qu'on injecte également l'encodage positionnel (RoPE).
3.  **L'empilement des Blocs (Le cerveau)** : Comme l'illustre la **Figure 3-20**, le vecteur de chaque token traverse la pile de blocs Transformer (souvent 12, 24 ou même 96 couches). 

{{< bookfig src="58.png" week="03" >}}

{{% hint info %}}
🔑 **Note technique** : Dans un décodeur (GPT/Llama), chaque token possède son propre "flux" ou "stream" de calcul. Ils montent les étages de la pile en parallèle, mais ils ne peuvent regarder que vers le bas (les tokens précédents) grâce au masquage d'attention.
{{% /hint %}}

4.  **Le vecteur final** : À la sortie du dernier bloc, nous obtenons un nouveau vecteur pour chaque token d'entrée. Mais pour la génération, seul le vecteur du **dernier token** nous intéresse. Pourquoi ? Parce que c'est lui qui contient la synthèse de tout le contexte nécessaire pour prédire la suite.

## 2. La Language Modeling Head (LM Head)

Le vecteur qui sort de la pile est un objet mathématique abstrait de dimension 768 ou 4096. Comment le transformer en un mot humain ? 

C'est le rôle de la **Figure 3-21 — LM Head**. 

{{< bookfig src="59.png" week="03" >}}

*   **Projection Linéaire** : On multiplie ce vecteur final par une immense matrice qui le projette dans un espace dont la taille est égale à celle de votre vocabulaire (ex: 50 257 dimensions).
*   **Les Logits** : Nous obtenons des scores bruts appelés **logits**. Un logit élevé pour l'index "42" signifie que le modèle pense très fort que le mot "pomme" est le suivant.
*   **Softmax** : On applique une fonction Softmax pour transformer ces scores en probabilités réelles entre 0 et 1. 

{{% hint info %}}
🔑 **La distinction du Professeur Henni** : Le modèle ne choisit pas un mot. Il calcule une météo de probabilités sur tout le dictionnaire. C'est la stratégie de décodage (Sampling) qui choisira ensuite l'élu parmi les plus probables.
{{% /hint %}}

## 3. Le secret de la vitesse : Le Cache KV

{{% hint warning %}}
**Attention : erreur fréquente ici !** Si vous ne comprenez pas le Cache KV, vous ne comprendrez jamais pourquoi les LLM coûtent si cher à faire tourner.
{{% /hint %}}

Le processus de génération est **autorégressif**. Pour générer une phrase de 10 mots, le modèle doit faire 10 forward passes complets. 
*   Passage 1 : Entrée "Le", prédit "chat".
*   Passage 2 : Entrée "Le chat", prédit "mange".
*   Passage 3 : Entrée "Le chat mange", prédit "la"...

Problème : à chaque étape, le modèle doit recalculer l'attention pour les mots qu'il a déjà traités ! C'est un gaspillage monumental. 

{{% hint info %}}
🔑 **La solution : Le Cache KV (Key-Value Cache)**. 
Regardez la **Figure 3-22**. L'idée est de stocker dans la mémoire VRAM du GPU les vecteurs **Key** et **Value** de tous les tokens passés.
{{% /hint %}}

{{< bookfig src="63.png" week="03" >}}

**Analogie** : Imaginez un chef cuisinier qui prépare un repas complexe étape par étape. Au lieu de refaire la sauce à chaque fois qu'il ajoute un nouvel ingrédient dans l'assiette, il garde la sauce prête dans un bol sur le côté. Le Cache KV, c'est ce bol. Le modèle n'a plus qu'à calculer la **Query** du nouveau mot et à la comparer aux **Keys** et **Values** déjà en mémoire.

**Impact sur la performance** : 
Sans cache KV, le temps de génération augmente de façon quadratique avec la longueur du texte. Avec le cache, il devient linéaire. Comme vous le verrez dans l'exercice de laboratoire, l'activation du cache peut diviser le temps de réponse par 5 ou 10 !

## 4. Analyse de structure : Regarder sous le capot

Pour finir, je veux que vous appreniez à lire la carte d'identité d'un modèle. En utilisant la bibliothèque `transformers`, nous pouvons imprimer la structure exacte d'un LLM. 

```python
# Installation : pip install transformers
from transformers import AutoModelForCausalLM

# Utilisons TinyLlama (modèle très léger, parfait pour l'analyse)
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

print(model)
```
<!-- TODO: add colab link -->

{{% hint warning %}}
En exécutant ce code, vous verrez apparaître des termes comme `LlamaDecoderLayer`, `LlamaAttention`, `LlamaRMSNorm`. 🔑 **C'est non-négociable :** vous devez être capables de reconnaître dans cet affichage informatique les blocs théoriques que nous avons étudiés cette semaine. La `LlamaMLP` est votre réseau Feedforward, et le `lm_head` final est votre traducteur vecteur-vers-mots.
{{% /hint %}}

## 5. Éthique : Le coût de la mémoire

{{% hint danger %}}
« Mes chers étudiants, le Cache KV est une bénédiction pour la vitesse, mais c'est un fardeau pour les ressources. »

Le cache KV consomme une quantité massive de mémoire vive sur le GPU (VRAM). 
*   **Inégalité matérielle** : Un modèle avec une fenêtre de contexte de 128 000 tokens nécessite des dizaines de gigaoctets de cache KV. Cela signifie que l'IA de pointe devient inaccessible pour ceux qui n'ont pas de serveurs surpuissants.
*   **Consommation électrique** : Maintenir ces données en cache et multiplier les accès mémoire a un coût énergétique. 

🔑 **C'est votre défi :** En tant qu'experts, vous devrez arbitrer entre la vitesse de réponse (meilleure expérience utilisateur) et la consommation de mémoire. Des techniques comme la **quantification du cache KV** (réduire la précision des vecteurs stockés) sont les nouvelles frontières d'une IA plus sobre et plus accessible.
{{% /hint %}}

## Synthèse de la Semaine 3

Quel voyage passionnant !
1.  Nous avons appris à calculer les **scores d'attention** (Q, K, V).
2.  Nous avons donné un sens de l'ordre au modèle via les **rotations positionnelles (RoPE)**.
3.  Nous avons stabilisé les calculs grâce aux **blocs et aux normalisations**.
4.  Nous avons optimisé la production de texte avec le **Cache KV**.

Vous avez maintenant une compréhension "moteur" complète. Vous ne voyez plus les LLM comme de la magie, mais comme une série de multiplications matricielles extraordinairement bien orchestrées. La semaine prochaine, nous allons enfin sortir du laboratoire pour utiliser ces modèles sur des tâches concrètes : nous étudierons les **Modèles de Représentation (Encoder-only)** comme BERT pour classer et comprendre le monde !
