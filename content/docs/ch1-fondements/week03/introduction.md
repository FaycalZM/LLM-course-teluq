---
title: "Introduction"
weight: 1
---

Bonjour à toutes et à tous ! J'espère que vous avez bien dormi, car aujourd'hui, nous entrons dans la "salle des machines". ⚙️ La semaine dernière, nous avons étudié les atomes (les tokens) et leur position dans l'espace (les embeddings). Aujourd'hui, nous allons voir comment ces atomes interagissent pour créer de la pensée artificielle. Nous allons disséquer le mécanisme d'attention, non plus seulement comme une intuition, mais comme une symphonie mathématique de haute précision.

{{% hint info %}}
🔑 **Je dois insister :** ce que nous allons voir — l'équation de la Scaled Dot-Product Attention — est le secret le mieux gardé de la révolution technologique actuelle. Prenez votre souffle, nous allons rendre l'invisible visible.
{{% /hint %}}

**Rappel semaine précédente** : La semaine dernière, nous avons exploré la tokenisation et les embeddings, comprenant comment le texte est converti en vecteurs denses et comment les modèles comme BERT créent des représentations contextuelles.

## Objectifs de la semaine

À la fin de cette semaine, vous saurez :
*   Expliquer et calculer mathématiquement le mécanisme de self-attention (Q, K, V).
*   Comprendre l'importance de l'encodage positionnel rotatif (RoPE).
*   Décortiquer la structure d'un bloc Transformer moderne (Norm, Residuals, MLP).
*   Analyser le passage de l'information (Forward Pass) et l'optimisation par cache KV.
