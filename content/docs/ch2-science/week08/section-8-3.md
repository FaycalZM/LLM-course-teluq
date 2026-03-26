---
title: "8.3 Raisonnement et structuration"
weight: 4
---

## Dépasser l'intuition : Quand l'IA prend le temps de réfléchir
Bonjour à toutes et à tous ! J'espère que vous êtes bien accrochés, car nous abordons aujourd'hui le "Saint Graal" du Prompt Engineering. Jusqu'ici, nous avons traité nos modèles comme des dictionnaires ou des traducteurs rapides (Système 1). Mais que se passe-t-il quand nous leur demandons de résoudre un problème de mathématiques complexe, de déjouer un paradoxe logique ou de planifier une stratégie d'entreprise ? 
> [!IMPORTANT]
🔑 **Je dois insister :** un LLM, par défaut, est un parieur statistique. Il veut donner la réponse la plus probable *immédiatement*. 

> Aujourd'hui, nous allons lui apprendre à ralentir, à sortir son "brouillon mental" et à raisonner étape par étape. Bienvenue dans le monde du **Raisonnement Augmenté**.


## La distinction entre Système 1 et Système 2
Pour comprendre l'intérêt de la structuration du raisonnement, nous devons faire un détour par la psychologie cognitive, citée par *Daniel Kahneman* et reprise dans le domaine des LLM.
*   **Système 1 (Pensée Rapide)** : Intuition, réflexe, génération immédiate. C'est le mode par défaut du LLM qui prédit le token suivant sans "réfléchir".
*   **Système 2 (Pensée Lente)** : Logique, calcul, vérification, décomposition. C'est ce que nous essayons de déclencher via les techniques que nous allons voir.

> [!NOTE]
🔑 **Mon intuition :** Imaginez que je vous demande : "Combien font 17 fois 24 ?". Si vous répondez au hasard, c'est le Système 1. Si vous prenez un papier et un crayon pour poser l'opération, c'est le Système 2. Les techniques de raisonnement sont le "papier et le crayon" du LLM.


## Chain-of-Thought (CoT) : La puissance du brouillon
L'article fondateur de Wei et al. (2022) a introduit le **Chain-of-Thought (Chaîne de pensée)**. Cette technique est illustrée par la **Figure 8-8 : Le Chain-of-Thought par l'exemple**.

{{< bookfig src="148.png" week="08" >}}

Décortiquons le contenu de cette figure capitale :
*   **Le problème** : La figure montre à gauche un prompt "standard" (Few-shot classique). On donne une question mathématique et une réponse directe. Résultat ? Le modèle se trompe sur la nouvelle question car il essaie de deviner le chiffre final d'un coup.
*   **La solution CoT** : À droite, l'exemple fourni au modèle inclut le **cheminement logique**. "Roger a 5 balles. Il achète 2 boîtes de 3, donc 2x3=6. 5+6=11." 
*   **L'effet** : En voyant ce modèle de pensée, le Modèle de Langage imite cette structure pour la nouvelle question. Il décompose : "La cafétéria avait 23 pommes. Elles en ont utilisé 20, il en reste 3. Elles en achètent 6, donc 3+6=9."

> [!TIP]
🔑 **Pourquoi cela fonctionne-t-il techniquement ?** En forçant le modèle à générer les étapes de calcul avant la réponse finale, vous permettez au mécanisme d'attention (Semaine 3) de s'appuyer sur les tokens de raisonnement qu'il vient de s'écrire à lui-même. C'est une extension de la puissance de calcul via la génération de tokens intermédiaires.


## Le miracle du Zero-shot CoT : "Réfléchissons étape par étape"
Parfois, vous n'avez pas d'exemples sous la main pour faire du CoT. C'est là qu'intervient la découverte de Kojima et al. (2022), illustrée par la **Figure 8-9 : Chain-of-Thought sans exemple**.

{{< bookfig src="149.png" week="08" >}}

La figure montre qu'en ajoutant simplement la phrase magique **"Let's think step by step"** (Réfléchissons étape par étape) à la fin d'une question complexe, le modèle change de comportement. 
*   **Avant** : Il donne une réponse brute (souvent fausse).
*   **Après** : Il s'auto-conditionne à décomposer sa réponse en paragraphes logiques. 

> [!WARNING]
⚠️ **Attention : erreur fréquente ici !** Penser que cette phrase résout tout. 

> Sur des modèles de petite taille (moins de 7B paramètres), le Zero-shot CoT peut parfois aggraver les choses en faisant "délirer" le modèle sur de fausses pistes logiques. 

> [!IMPORTANT]
🔑 **Je dois insister :** Le raisonnement est une capacité émergente qui dépend de la taille et de la qualité du pré-entraînement du modèle.


## Self-Consistency : La sagesse de la majorité
Un seul chemin de raisonnement peut être erroné. Pour stabiliser les résultats, nous utilisons la **Self-Consistency** (Auto-cohérence), représentée en **Figure 8-10** .

{{< bookfig src="150.png" week="08" >}}

L'idée est inspirée des méthodes d'ensemble en Machine Learning :
1.  On pose la question au modèle avec un prompt CoT.
2.  On génère non pas une, mais **plusieurs réponses** (par exemple 5 ou 10) en utilisant une température élevée (ex: 0.7).
3.  Chaque réponse aura un chemin de raisonnement potentiellement différent.
4.  On regarde quelle réponse finale (le chiffre ou la conclusion) apparaît le plus souvent.

> [!TIP]
🔑 **L'intuition technique :** Le modèle peut faire une erreur de calcul dans un chemin, mais il est statistiquement improbable qu'il fasse la *même* erreur exacte dans 5 chemins différents. La majorité l'emporte et la fiabilité s'envole.


## Tree-of-Thought (ToT) : L'exploration par l'arbre des possibles
Pour les problèmes vraiment complexes (ex: écrire un plan marketing complet ou résoudre une énigme de type Sudoku), le CoT linéaire ne suffit plus. Il faut explorer plusieurs idées et pouvoir revenir en arrière. C'est le **Tree-of-Thought**, illustré par la **Figure 8-11** .

{{< bookfig src="151.png" week="08" >}}

Cette architecture, beaucoup plus lourde, traite le raisonnement comme une recherche dans un arbre :
*   **Génération de pensées** : Le modèle propose 3 idées pour l'étape 1.
*   **Évaluation** : Le modèle (ou un autre modèle) note la pertinence de chaque idée. 
*   **Élagage (Pruning)** : On abandonne les branches qui mènent à une impasse.
*   **Backtracking** : Si aucune branche ne marche, on remonte à l'étape précédente pour essayer autre chose.

> [!NOTE]
🔑 **Le concept d'Expert Discussion (Figure 8-12)** : Une variante simplifiée consiste à demander au modèle d'imaginer trois experts (ex: un ingénieur, un designer, un juriste) qui débattent du problème dans le prompt. Le consensus final est souvent bien plus robuste qu'une réponse unique.
<a id="fig-8-12"></a>

{{< bookfig src="152.png" week="08" >}}

## Structuration de la sortie : Forcer la rigueur
Le raisonnement ne sert à rien si le résultat est noyé dans un texte illisible. La structuration consiste à imposer une forme logique à la pensée de l'IA. 
Nous utilisons pour cela des techniques de **Contraintes de format** (Format constraints) :
*   **Markdown** : Pour les titres et les listes.
*   **Balises de raisonnement** : Utiliser `<thinking>...</thinking>` pour séparer le brouillon de la réponse finale `<answer>...</answer>`.


## Laboratoire de code : Implémentation CoT et Vote majoritaire (Colab T4)
Voici comment orchestrer un raisonnement Système 2 avec Phi-3-mini. Nous allons résoudre un problème de logique en simulant une Self-Consistency simplifiée.

```python
# Testé sur Colab T4 16GB VRAM
from transformers import pipeline
import torch
from collections import Counter

model_id = "microsoft/Phi-3-mini-4k-instruct"
pipe = pipeline("text-generation", model=model_id, device_map="auto", torch_dtype=torch.float16)

# --- PROMPT AVEC CHAIN-OF-THOUGHT ---
puzzle = "If I have 3 baskets with 5 apples each, and I give 2 apples to a friend, but then find another basket with 4 apples, how many apples do I have?"

cot_prompt = f"""Question: {puzzle}
Answer: Let's think step by step:
1."""

# --- RÉPONSE : GÉNÉRATION MULTIPLE (SELF-CONSISTENCY) ---
print("Génération des chemins de pensée...")
responses = pipe(
    cot_prompt, 
    max_new_tokens=150, 
    num_return_sequences=3, # On génère 3 versions
    do_sample=True, 
    temperature=0.7
)

final_answers = []
for i, res in enumerate(responses):
    text = res['generated_text']
    print(f"\n--- Chemin {i+1} ---\n{text}")
    # Extraction simplifiée du dernier nombre (logique d'exemple)
    import re
    nums = re.findall(r'\d+', text)
    if nums: final_answers.append(nums[-1])

# Vote majoritaire
if final_answers:
    majority = Counter(final_answers).most_common(1)[0][0]
    print(f"\n🔑 RÉPONSE FINALE VALIDÉE : {majority}")
```

> [!WARNING]
⚠️ Observez les chemins générés. Parfois, un chemin sera très verbeux et un autre très sec. C'est la beauté (et le danger) du décodage probabiliste. En tant qu'ingénieurs, votre rôle est de "contenir" cette variation par des prompts de structuration.


## Éthique et Responsabilité : Le piège de la rationalisation

> [!CAUTION]
⚠️ Mes chers étudiants, méfiez-vous de la "belle parole".

Un LLM peut produire un raisonnement étape par étape qui semble parfait, mais dont la conclusion est fausse. Ou pire : il peut avoir la bonne réponse par chance, et inventer un raisonnement (rationalisation) pour la justifier après coup.
1.  **L'illusion de logique** : Ce n'est pas parce qu'un modèle écrit "D'après la loi de Newton..." qu'il applique réellement les principes de la physique. Il corrèle des mots de physique.
2.  **Biais de confirmation** : Si votre prompt contient une erreur ("Pourquoi 2+2 font 5 ?"), le modèle, dans son désir d'être "utile" ([**Sycophancy, Semaine 5**]({{< relref "section-5-2.md" >}}#sycophancy)), inventera une logique absurde pour valider votre erreur. 

> [!IMPORTANT]
🔑 **Je dois insister :** Le Chain-of-Thought n'est pas une preuve de vérité, c'est une aide à la performance. Vous restez l'ultime arbitre de la logique. Utilisez les modèles d'experts (ToT) pour confronter les points de vue et limiter ces biais.

---
Vous avez maintenant les clés du raisonnement. Vous ne demandez plus seulement à l'IA de parler, vous lui demandez de construire une pensée. C'est une étape gigantesque vers des applications industrielles robustes. Mais attention : une pensée brillante dans un format brouillon est difficile à exploiter. Dans la prochaine section ➡️, nous allons apprendre à "verrouiller" la sortie pour qu'elle respecte une grammaire stricte, comme le JSON, indispensable pour connecter vos IA à vos logiciels.