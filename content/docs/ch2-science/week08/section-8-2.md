---
title: "8.2 Techniques avancées"
weight: 3
---

## L'apprentissage sans mise à jour : Le miracle de l'In-Context Learning
Bonjour à toutes et à tous ! Nous entrons maintenant dans la partie la plus spectaculaire de l'ingénierie des prompts. La semaine dernière, nous avons vu que pour spécialiser un modèle BERT, il fallait modifier ses poids (le fine-tuning). Mais avec les modèles génératifs comme GPT ou Phi-3, il existe une forme de magie appelée l'**In-Context Learning** (Apprentissage en contexte). 

> [!IMPORTANT]
🔑 **Je dois insister :** le modèle n'apprend rien de nouveau de manière permanente, ses neurones ne changent pas. Il utilise simplement sa mémoire de travail pour s'adapter à vos exemples. C'est comme donner une fiche de consignes à un intérimaire très intelligent : il comprend instantanément ce qu'il doit faire pour la durée de sa mission.


## La hiérarchie du guidage : De Zero-shot à Few-shot
Comme l'illustre la **Figure 8-6 : Zero-shot, one-shot et few-shot prompting**, nous disposons d'une échelle de précision pour guider le modèle. 

{{< bookfig src="146.png" week="08" >}}

Décortiquons cette figure ensemble : elle représente une tâche de classification de sentiments ("neutral", "negative", "positive") et montre comment l'ajout d'exemples transforme la réponse de l'IA. 

### 1. Zero-shot Prompting (Zéro exemple)
C'est ce que nous avons fait jusqu'ici. On donne une instruction brute ("Classifie ce texte"). 
*   **Pourquoi ça marche ?** Le modèle s'appuie uniquement sur ce qu'il a appris durant son pré-entraînement massif.
*   **Le risque** : Le modèle peut ne pas respecter le format de sortie ou mal interpréter une nuance très spécifique à votre métier. 

> [!WARNING]
⚠️ **Attention : erreur fréquente ici !** Penser que le Zero-shot suffit pour des tâches complexes. Si le modèle échoue, ne changez pas de modèle tout de suite : passez au One-shot.

### 2. One-shot Prompting (Un exemple)
On fournit un seul couple "Entrée / Sortie" pour servir de modèle. 
*   **Analyse de la Figure 8-6 (milieu)** : En montrant au modèle que pour la phrase "L'hôtel était correct", la réponse attendue est "Neutral", vous fixez non seulement la logique, mais aussi le vocabulaire autorisé.
*   **L'effet miroir** : Le modèle va imiter votre ton, votre concision et votre formatage. C'est l'outil idéal pour fixer un style d'écriture.

### 3. Few-shot Prompting (Plusieurs exemples)

> [!TIP]
🔑 **C'est la technique la plus robuste.** On donne généralement entre 3 et 8 exemples au modèle. 
*   **Analyse de la Figure 8-6 (droite)** : En listant plusieurs exemples variés (un positif, un négatif, un neutre), vous créez une "mini-base de connaissances" temporaire. 
*   **Le cas "Gigamuru"** : C'est un exemple fascinant. On invente un mot, le "Gigamuru" (un instrument de musique japonais imaginaire). En donnant un exemple de phrase utilisant ce mot, le modèle est capable de générer de nouvelles phrases parfaites. Cela prouve que le modèle peut apprendre des concepts totalement nouveaux... tant qu'ils restent dans son prompt ! 


## L'art de choisir ses exemples (Few-shot engineering)

> [!WARNING]
⚠️ Ne jetez pas n'importe quels exemples dans votre prompt ! 

La qualité du Few-shot dépend de trois règles non-négociables :
1.  **La Diversité** : Si vous ne donnez que des exemples de critiques positives, le modèle aura un biais optimiste. Donnez des exemples qui couvrent tous les cas de figure.
2.  **L'Ordre des exemples** : Les modèles souffrent de "récence". L'exemple situé juste avant la question réelle a plus d'influence que le premier. 
>> [!TIP]
🔑 **Mon conseil** : Mettez votre exemple le plus complexe ou le plus important en dernier dans la liste.
3.  **La Cohérence du format** : Si votre premier exemple est en JSON et le second en texte brut, le modèle sera perdu. Soyez un métronome dans votre structure.


## Chain Prompting : Diviser pour mieux régner
Parfois, une tâche est trop lourde pour être résolue en une seule fois. Imaginez demander à un architecte de dessiner les plans, de commander les matériaux et de construire la maison en une seule phrase. C'est la recette du désastre.

Pour les LLM, nous utilisons le **Chain Prompting** (Prompting en chaîne), illustré magnifiquement par la **Figure 8-7 : Utilisation de chaînes de prompts**.

{{< bookfig src="147.png" week="08" >}}

Cette figure décrit le processus de création d'un produit :
1.  **Lien 1** : Créer un nom de produit à partir de caractéristiques.
2.  **Lien 2** : Créer un slogan à partir du nom trouvé.
3.  **Lien 3** : Écrire un argumentaire de vente à partir du nom et du slogan.

> [!IMPORTANT]
🔑 **Je dois insister sur l'avantage technique :** En découpant la tâche, vous permettez au modèle d'accorder 100% de son "attention" ([**Semaine 3**]({{< relref "section-3-1.md" >}})) à un petit problème à la fois. Cela réduit drastiquement les **hallucinations**. Le modèle n'a plus besoin de jongler avec 5 contraintes en même temps ; il se concentre sur une seule brique.

## Prompt Chaining vs Sequential Prompting
Dans le Sequential Prompting, nous faisons plusieurs appels au modèle. La sortie du prompt A devient une variable dans le prompt B. 
*   *Analogie* : C'est comme une ligne de montage à l'usine. Chaque étape affine la pièce précédente.
*   *Usage professionnel (exemple)* : Traduire un texte (Prompt 1), puis demander à l'IA de corriger les erreurs culturelles de la traduction (Prompt 2), puis formater le résultat en HTML (Prompt 3).

## Laboratoire de code : Implémentation Few-shot et Chaining (Colab T4)
Voici comment structurer une chaîne de prompts sophistiquée en Python. Nous allons utiliser Phi-3-mini pour simuler un pipeline de création de contenu.

```python
# Testé sur Colab T4 16GB VRAM
# !pip install transformers accelerate

from transformers import pipeline
import torch

# Initialisation du modèle léger
model_id = "microsoft/Phi-3-mini-4k-instruct"
pipe = pipeline("text-generation", model=model_id, device_map="auto", torch_dtype=torch.float16)

# --- ÉTAPE 1 : FEW-SHOT POUR L'EXTRACTION ---
# On montre au modèle comment extraire des entités d'un texte complexe
extraction_prompt = [
    {"role": "user", "content": "Extract items from: I bought an apple, a car, and a house."},
    {"role": "assistant", "content": "1. apple\n2. car\n3. house"},
    {"role": "user", "content": "Extract items from: My inventory includes a sword, a shield, and 5 potions."}
]

# 1. Exécution du premier maillon (Extraction)
res1 = pipe(extraction_prompt, max_new_tokens=50)
extracted_items = res1[0]['generated_text'][-1]['content']

print(f"Maillon 1 (Objets) :\n{extracted_items}")

# 2. Exécution du second maillon (Chaining)
# On utilise la sortie du maillon 1 pour construire le prompt 2
story_prompt = [
    {"role": "system", "content": "You are a fantasy writer."},
    {"role": "user", "content": f"Write a 2-sentence story using these items: {extracted_items}"}
]

res2 = pipe(story_prompt, max_new_tokens=100)
print(f"\nMaillon 2 (Histoire) :\n{res2[0]['generated_text'][-1]['content']}")
```

## Le danger de la "Paresse de Contexte"

> [!WARNING]
⚠️ Ne surchargez pas votre prompt d'exemples inutiles.

Chaque mot que vous ajoutez au prompt consomme des tokens et réduit l'espace disponible pour la réponse. De plus, si vos exemples sont trop similaires ("Un chat est un animal", "Un chien est un animal"), vous n'apprenez rien de nouveau au modèle, vous gaspillez simplement de la mémoire. 

> [!TIP]
🔑 **La règle d'or** : 3 exemples contrastés (un cas simple, un cas complexe, un cas limite) valent mieux que 20 exemples répétitifs.


## Éthique et Responsabilité : Le modèle "perroquet"

> [!CAUTION]
⚠️ Mes chers étudiants, le Few-shot peut être une prison sémantique.

Si vous donnez au modèle des exemples qui contiennent des stéréotypes, il va les reproduire avec une fidélité effrayante. Par exemple, si dans vos exemples de classification de CV, vous ne montrez que des hommes pour des rôles techniques, le modèle "apprendra" en quelques millisecondes que c'est le pattern à suivre. 

> [!IMPORTANT]
🔑 **Conséquence éthique :** Le Few-shot est une injection directe de biais. En tant qu'experts, vous devez auditer vos exemples d'entraînement avec autant de soin que si vous écriviez le code source de l'application. Vous êtes les "enseignants" de l'IA le temps d'un prompt.

---
Vous maîtrisez maintenant les techniques de guidage par l'exemple et la division des tâches. Vous savez comment transformer une IA hésitante en un exécutant précis. Mais comment s'assurer que le modèle "réfléchit" vraiment avant de répondre à une énigme ? Dans la prochaine section ➡️, nous allons explorer les techniques de **Raisonnement**, comme le célèbre **Chain-of-Thought**, pour donner aux LLM une véritable "conscience" étape par étape.