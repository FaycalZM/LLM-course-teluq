---
title: "3.1 Le mécanisme d'attention : Mathématiques détaillées"
weight: 2
---

{{< katex />}}

## La bibliothèque infinie : L'analogie Query, Key, Value

Avant de plonger dans les matrices, laissez-moi vous raconter une histoire. Imaginez que vous soyez dans une bibliothèque immense à la recherche d'une information précise sur le "climat". 

1.  **La Query (Requête - $Q$)** : C'est ce que vous avez en tête, votre intention. Vous marchez dans les allées en criant : "Je cherche des infos sur le climat !".
2.  **La Key (Clé - $K$)** : C'est l'étiquette collée sur le dos de chaque livre. Un livre a pour étiquette "Météo", un autre "Cuisine", un autre "Écologie".
3.  **La Value (Valeur - $V$)** : C'est le contenu réel à l'intérieur du livre. 

Le mécanisme d'attention est le processus par lequel vous comparez votre **Query** à toutes les **Keys** de la bibliothèque. Si votre Query ("Climat") ressemble beaucoup à une Key ("Météo"), vous allez accorder beaucoup d'importance à la **Value** de ce livre. Si la Key est "Cuisine", vous l'ignorerez.

{{% hint info %}}
🔑 **C'est le cœur de l'attention :** extraire l'information pertinente en comparant des intentions à des étiquettes.
{{% /hint %}}

## Les quatre étapes du calcul de l'attention

Comme illustré dans les **Figures 3-1 à 3-7**, le calcul se décompose en étapes mathématiques rigoureuses que tout expert en LLM doit connaître par cœur.

{{< bookfig src="68.png" week="03" >}}
{{< bookfig src="69.png" week="03" >}}
{{< bookfig src="70.png" week="03" >}}
{{< bookfig src="71.png" week="03" >}}
{{< bookfig src="72.png" week="03" >}}
{{< bookfig src="73.png" week="03" >}}
{{< bookfig src="74.png" week="03" >}}

### Étape 1 : Les Projections Linéaires

Chaque embedding d'entrée ($x$) est multiplié par trois matrices de poids apprises ($W^Q, W^K, W^V$) pour générer nos trois vecteurs :
$$Q = x \cdot W^Q$$
$$K = x \cdot W^K$$
$$V = x \cdot W^V$$

{{% hint warning %}}
**Attention : erreur fréquente ici !** Les Query, Key et Value ne sont pas les embeddings d'origine. Ce sont des transformations de ces embeddings dans des espaces différents pour que le modèle puisse apprendre des relations complexes.
{{% /hint %}}

### Étape 2 : Le score de pertinence (Dot Product)

On calcule la similarité entre la Query du mot actuel et les Keys de tous les autres mots de la phrase via un produit scalaire ($Q \cdot K^T$). 
Plus le résultat est élevé, plus les deux mots sont "liés" sémantiquement. Par exemple, dans "Le chat mange", la Query de "mange" aura un produit scalaire très élevé avec la Key de "chat".

### Étape 3 : Le passage à l'échelle (Scaling) et Softmax

C'est ici qu'intervient la précision d'ingénierie. On divise le score par la racine carrée de la dimension des vecteurs clés ($\sqrt{d_k}$). 

{{% hint info %}}
🔑 **Je dois insister :** Pourquoi ce $\sqrt{d_k}$ ? Parce que sans lui, pour des vecteurs de grande taille, les scores explosent et le gradient disparaît lors de l'entraînement, rendant le modèle incapable d'apprendre. On applique ensuite une fonction **Softmax** pour transformer ces scores en probabilités (dont la somme fait 1).
{{% /hint %}}

### Étape 4 : L'agrégation finale

On multiplie chaque vecteur **Value** par son score de probabilité et on additionne le tout. Le résultat est un nouvel embedding pour notre mot, mais un embedding qui a "absorbé" la substance de ses voisins utiles.

## L'équation sacrée du Transformer

Tout ce processus est résumé dans cette formule unique, que vous devriez être capables de réciter :
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

## Multi-Head Attention : L'intelligence parallèle

Mais attendez, pourquoi n'utiliser qu'une seule paire de lunettes ? Le Transformer utilise la **Multi-Head Attention (Attention à têtes multiples)**. 

Comme vous le voyez sur la **Figure 3-8**, le modèle divise ses vecteurs en plusieurs "têtes". 
*   La tête 1 peut se concentrer sur les relations sujet-verbe.
*   La tête 2 sur les adjectifs et les noms.
*   La tête 3 sur les références temporelles.

{{< bookfig src="79.png" week="03" >}}

{{% hint info %}}
🔑 **L'avantage est colossal :** cela permet au modèle de comprendre simultanément plusieurs aspects d'une même phrase. C'est comme si dix experts lisaient la phrase en même temps et mettaient leurs notes en commun à la fin.
{{% /hint %}}

## Exemple numérique pas à pas

Pour bien ancrer la théorie, faisons un calcul simplifié avec des dimensions minuscules.
Imaginons un mot avec un embedding Query $q = [1, 0]$ et deux mots voisins avec des Keys $k_1 = [1, 0]$ (très similaire) et $k_2 = [0, 1]$ (très différent). Supposons $d_k=1$ pour simplifier.

1.  **Scores (Dot product)** : 
    *   $q \cdot k_1 = (1\times1) + (0\times0) = 1$
    *   $q \cdot k_2 = (1\times0) + (0\times1) = 0$
2.  **Softmax** : 
    *   $\text{Score } 1 = \frac{e^1}{e^1 + e^0} \approx 0.73$
    *   $\text{Score } 2 = \frac{e^0}{e^1 + e^0} \approx 0.27$
3.  **Résultat** : Le nouvel embedding sera composé à 73% de la Value du mot 1 et à 27% de la Value du mot 2.

Vous voyez ? La mathématique a littéralement "écouté" le mot 1 au détriment du mot 2.

## Optimisations modernes : FlashAttention et GQA

{{% hint warning %}}
En tant qu'ingénieurs, vous devez savoir que l'attention classique a un coût astronomique. Elle est en $O(L^2)$ : si vous doublez la longueur du texte, le temps de calcul est multiplié par quatre.
{{% /hint %}}

Pour résoudre cela, les modèles récents comme Llama-3 utilisent :
*   **Grouped-Query Attention (GQA)** : Illustré en **Figure 3-9**, où plusieurs Queries partagent les mêmes Keys et Values pour économiser de la mémoire VRAM.

{{< bookfig src="78.png" week="03" >}}

*   **FlashAttention** : Une réimplémentation du calcul au niveau du matériel (GPU) qui évite les allers-retours inutiles dans la mémoire, accélérant la génération de manière spectaculaire.

## Éthique : L'attention comme miroir

L'attention est un miroir. Si le modèle accorde une attention démesurée à des mots chargés de préjugés, c'est parce qu'il a appris que ces corrélations étaient "pertinentes" dans nos propres écrits. 

{{% hint danger %}}
🔑 **Je dois insister :** l'attention mathématique n'est pas une attention morale. Elle ne distingue pas le fait de la fiction, ou le respect du mépris. Elle ne voit que des poids statistiques. C'est à nous, par le fine-tuning (Semaine 12), de lui apprendre quelle attention est souhaitable pour une société juste.
{{% /hint %}}

## Extrait de Code : Visualiser les têtes d'attention

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
```
<!-- TODO: add colab link -->

{{% hint info %}}
🔑 **L'intuition finale** : L'attention a tué la récurrence parce qu'elle a permis au langage d'être traité comme une structure spatiale globale plutôt que comme une corvée temporelle.
{{% /hint %}}

Reprenez votre souffle. Nous venons de voir comment les mots se parlent. Dans la section suivante, nous allons voir comment ils se situent dans l'espace grâce à l'encodage positionnel.
