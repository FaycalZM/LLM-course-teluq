---
title: "5.2 Evolution des modèles"
weight: 3
---

## De l'autocomplétion à la conversation : Le grand saut
Bonjour à nouveau ! Avez-vous déjà essayé de discuter avec un dictionnaire ? C'est frustrant, n'est-ce pas ? C'est pourtant ce que nous faisions au début avec les modèles de fondation. Aujourd'hui, nous allons comprendre comment nous avons transformé ces "encyclopédies statistiques" en partenaires de discussion capables de nous aider à coder, à écrire des poèmes ou à planifier nos vacances.

L'évolution des LLM ne s'est pas faite uniquement par l'augmentation du nombre de neurones (paramètres), mais par un changement radical dans la manière dont nous les "éduquons". Comme l'illustre la [**Figure 1-26 : Modèles instruct/chat**]({{< relref "section-5-1.md" >}}#fig-5-3), nous sommes passés d'un modèle qui complète mécaniquement une phrase à un modèle qui comprend l'intention derrière une consigne.

> [!IMPORTANT]
🔑 **Je dois insister :** cette transition est ce qui a rendu l'IA accessible au grand public. Sans cette étape d'alignement, GPT ne serait resté qu'un outil de laboratoire pour chercheurs spécialisés. 

## Le Modèle de Fondation (Base Model) : L'érudit brut
Le point de départ de tout LLM moderne est le **Base Model** (Modèle de base). C'est un modèle qui a "lu" une part immense d'Internet. Sa seule mission est statistique : prédire le mot suivant. 
*   **Son comportement** : Si vous lui écrivez "Comment faire une omelette ?", il pourrait répondre par "Ingrédients : 3 œufs, sel, poivre..." mais il pourrait aussi répondre par "2. Comment faire des crêpes ? 3. Comment faire un gâteau ?", car il a appris que les listes de questions se suivent souvent. 
*   **Son utilité** : C'est une base de connaissances brute. Il n'a aucune notion de politesse, de sécurité ou de formatage. Il est le socle sur lequel nous allons bâtir l'intelligence de l'assistant.

## La métamorphose : SFT et Alignement
Pour transformer ce géant brut en un assistant comme ChatGPT, nous suivons un pipeline d'entraînement sophistiqué.

### 1. Supervised Fine-Tuning (SFT) / Instruction Tuning
On engage des humains pour écrire des milliers de dialogues parfaits. "Voici une question -> Voici la réponse idéale". On ré-entraîne le modèle de base sur ces exemples. C'est l'**Instruction Tuning**. Le modèle apprend enfin qu'une question appelle une réponse et non une autre question

<a id="RLHF"></a>

### 2. L'alignement par les préférences humaines (RLHF)
« Mais comment apprendre à une machine ce qui est "mieux" ou "plus poli" ? » C'est là qu'intervient le **Reinforcement Learning from Human Feedback (RLHF)**: 
*   On demande au modèle de générer deux réponses différentes pour la même question. 
*   Un humain choisit la meilleure (plus utile, plus sûre, plus claire). 
*   Un "modèle de récompense" (*Reward Model*) apprend à prédire ce que l'humain préfère. 
*   Enfin, le LLM est optimisé pour maximiser cette récompense. 

> [!NOTE]
🔑 Le RLHF n'ajoute pas de nouvelles connaissances au modèle ; il lui apprend à *mieux présenter* ce qu'il sait déjà et à rejeter les demandes dangereuses. C'est l'étape de la "civilité numérique".

## L'explosion de l'Open-Source : Reprendre le contrôle
Pendant longtemps, les LLM puissants étaient enfermés derrière les API d'OpenAI ou de Google (Modèles propriétaires). Mais tout a changé avec l'arrivée des **modèles open-source** (voir la **Figure 5-5 : Sélection de foundation models**). Des modèles comme **Llama-2/3** (Meta), **Mistral 7B** et la famille **Phi** (Microsoft) ont prouvé que l'on pouvait avoir des performances incroyables avec des modèles plus petits et ouverts

{{< bookfig src="134.png" week="05" >}}


### Le phénomène Phi-3-mini

> [!IMPORTANT]
🔑 **Je dois insister sur ce point technique :** La taille ne fait pas tout. Phi-3-mini, avec ses 3,8 milliards de paramètres, bat parfois des modèles deux fois plus gros. Pourquoi ? Parce que Microsoft l'a entraîné sur des données de "haute qualité" (des manuels scolaires générés par IA, très pédagogiques) plutôt que sur le "bruit" du web.

> [!TIP]
👍 Pour vos projets en laboratoire sur Colab, privilégiez toujours ces modèles compacts. Ils tiennent dans la mémoire de votre carte T4 et répondent presque instantanément.

## Mise en pratique : Utiliser un modèle de type "Chat"
Pour interagir avec un modèle de chat, nous devons respecter un format spécifique appelé **Chat Template**. Le modèle s'attend à voir des balises comme `<|user|>` et `<|assistant|>`. Hugging Face gère cela magnifiquement pour nous.

<a id="ex-chat-persona"></a>

```python
# Testé sur Colab T4 16GB VRAM
# !pip install transformers>=4.41.0 accelerate bitsandbytes

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

model_id = "microsoft/Phi-3-mini-4k-instruct"

# Chargement du modèle avec optimisation GPU
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto", 
    torch_dtype=torch.float16, # On réduit la précision pour économiser la VRAM
    trust_remote_code=False
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Création de la pipeline de discussion
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Structure d'un message pour un modèle de Chat
messages = [
    {"role": "system", "content": "You are a helpful science professor."},
    {"role": "user", "content": "Explique-moi pourquoi le ciel est bleu en une phrase."}
]

# Inférence
output = pipe(messages, max_new_tokens=50)
print(f"Assistant : {output[0]['generated_text'][-1]['content']}")

```

## Le Prompt Engineering : L'art de murmurer à l'oreille des modèles

> [!IMPORTANT]
🔑 **Notez bien cette distinction :** Dans un modèle de base, vous faites de l'autocomplétion. Dans un modèle Chat/Instruct, vous faites de l'**ingénierie de prompt**. 

Comme nous le verrons en Semaine 8, la façon dont vous définissez le "Persona" (ex: "Vous êtes un professeur de science") change radicalement la qualité de la réponse. C'est parce que le modèle, durant son alignement, a appris à reconnaître ces rôles.

## Éthique et Limites de l'Alignement

> [!CAUTION]
⚠️ **Mes chers étudiants, l'alignement est une arme à double tranchant.** 

>1.  **Le biais du censeur** : Si nous alignons trop un modèle pour qu'il soit "sûr", il peut devenir inutilement refusant (ex: refuser d'écrire une histoire triste par peur d'être "négatif"). 
>2.  **La complaisance (Sycophancy)** : Les modèles alignés ont tendance à toujours donner raison à l'utilisateur, même quand celui-ci se trompe. Si vous dites "Je pense que 2+2=5", un modèle trop aligné pourrait répondre "Vous avez raison, dans certains contextes...". 
>3.  **L'illusion d'intelligence** : Ne confondez pas la politesse et la fluidité avec la raison. Un modèle peut être parfaitement aligné, très poli, et vous donner un code informatique qui contient une faille de sécurité majeure. 

> [!TIP]
🔑 **Mon conseil** : Utilisez les versions "Chat" pour vos applications, mais gardez toujours un œil sur les versions "Base" pour comprendre la puissance brute du modèle sans les filtres comportementaux.

---
Nous avons maintenant vu comment les modèles ont évolué d'une simple boucle de prédiction à des assistants conversationnels alignés. Vous comprenez la différence entre le savoir brut et le comportement appris. Dans la section suivante, nous allons apprendre à régler les "boutons" de la machine : la **température** et le **top_p**, pour transformer une réponse robotique en une création inspirée.
