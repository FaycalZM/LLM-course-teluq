---
title: "Laboratoire "
weight: 6
---

Bonjour à toutes et à tous ! Nous passons maintenant à la phase de "sculpture". Dans ce laboratoire, vous n'êtes plus de simples utilisateurs, vous êtes des ingénieurs du contexte.

> [!IMPORTANT]
🔑 **Je dois insister :** l'IA est un miroir de votre clarté. Si votre prompt est flou, sa pensée le sera aussi. Nous allons apprendre à transformer des réponses banales en raisonnements brillants et à dompter l'imprévisibilité de la machine pour obtenir des données structurées. Prêt·e·s à murmurer à l'oreille des modèles ? C'est parti !


## 🔹 EXERCICE 1 : Optimisation de prompt par le raisonnement (Niveau 1)

**Objectif** : Transformer un prompt "Zero-shot" qui échoue en un prompt "Chain-of-Thought" réussi.

```python
# --- (QUESTION) ---
from transformers import pipeline
import torch

# Modèle léger TinyLlama pour Colab
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
pipe = pipeline("text-generation", model=model_id, torch_dtype=torch.bfloat16, device_map="auto")

# Le problème mathématique complexe
question = "The cafeteria had 23 apples. They used 20 for lunch and bought 6 more. How many apples do they have?"

# TÂCHE : Comparez la réponse directe et la réponse avec CoT
print("--- TEST 1 : RÉPONSE DIRECTE ---")
# [VOTRE CODE ICI]

```

<details>
<summary>Voir la réponse</summary>

```python
# --- RÉPONSE (CORRIGÉ) ---
# 1. Test sans raisonnement (le modèle risque de répondre 27 ou 29 par confusion)
prompt_direct = f"<|user|>\nQuestion: {question}\nAnswer:<|assistant|>\n"
res1 = pipe(prompt_direct, max_new_tokens=10, do_sample=False)
print(f"Direct: {res1[0]['generated_text'].split('Answer:')[-1].strip()}")

# 2. Test avec Zero-shot Chain-of-Thought
print("\n--- TEST 2 : RÉPONSE AVEC RAISONNEMENT (CoT) ---")
prompt_cot = f"<|user|>\nQuestion: {question}\nAnswer: Let's think step by step:<|assistant|>\n"
res2 = pipe(prompt_cot, max_new_tokens=100, do_sample=False)
print(f"CoT: {res2[0]['generated_text'].split('assistant|>')[-1].strip()}")
```

**Explications détaillées** :
*   **Résultats attendus** : Le test 1 donne souvent un chiffre faux. Le test 2 décompose : 23 - 20 = 3, puis 3 + 6 = 9.
*   **Justification** : En forçant le modèle à écrire les étapes de soustraction, on évite qu'il ne fasse une corrélation statistique trop rapide entre les nombres 23, 20 et 6.

</details>


---

## 🔹 EXERCICE 2 : Few-shot prompting pour l'extraction (Niveau 2)

**Objectif** : Apprendre au modèle un format d'extraction personnalisé et complexe sans fine-tuning.

```python
# --- CODE (QUESTION) ---
# On veut extraire le NOM et la COULEUR des fruits dans un format "Fruit: [Nom] | Color: [Couleur]"

messy_text = "I have a big red apple and a small yellow banana in my basket."

# TÂCHE : Construisez un prompt Few-shot avec 2 exemples.
```

<details>
<summary>Voir la réponse</summary>

```python
# --- RÉPONSE (CORRIGÉ) ---
few_shot_prompt = [
    {"role": "user", "content": "Extract: A green lime and an orange orange."},
    {"role": "assistant", "content": "Fruit: lime | Color: green\nFruit: orange | Color: orange"},
    {"role": "user", "content": f"Extract: {messy_text}"}
]

# Inférence
formatted_prompt = pipe.tokenizer.apply_chat_template(few_shot_prompt, tokenize=False, add_generation_prompt=True)
output = pipe(formatted_prompt, max_new_tokens=50, do_sample=False)

print("Résultat de l'extraction Few-shot :")
print(output[0]['generated_text'].split("<|assistant|>")[-1].strip())
```

**Explications détaillées** :
*   **Résultats attendus** : "Fruit: apple | Color: red / Fruit: banana | Color: yellow".
*   **Justification** : Le modèle identifie le pattern "Fruit: ... | Color: ..." grâce aux exemples et l'applique scrupuleusement au nouveau texte, même s'il contient des adjectifs perturbateurs ("big", "small").

</details>


---

## 🔹 EXERCICE 3 : Validation de sortie JSON (Niveau 3)

**Objectif** : Utiliser un prompt de structuration pour obtenir un objet JSON valide et le charger en Python.

```python
# --- CODE (QUESTION) ---
import json

# TÂCHE : Créez un prompt qui génère les statistiques d'un personnage de RPG.
# Le JSON doit avoir les clés : "name", "class", "power_level".

```
<details>
<summary>Voir la réponse</summary>

```python
# --- RÉPONSE (CORRIGÉ) ---
prompt_json = """<|user|>
Create a fantasy character profile. 
Output ONLY valid JSON code. No conversation.
Format:
{
  "name": "string",
  "class": "Warrior/Mage/Rogue",
  "power_level": int
}
<|assistant|>
"""

res = pipe(prompt_json, max_new_tokens=100, do_sample=False)
raw_content = res[0]['generated_text'].split("<|assistant|>")[-1].strip()

try:
    # Validation par chargement
    data = json.loads(raw_content)
    print("✅ JSON VALIDE GÉNÉRÉ :")
    print(json.dumps(data, indent=2))
except:
    print("❌ Erreur de formatage JSON.")
    print(f"Texte brut reçu : {raw_content}")

```

**Explications détaillées** :
*   **Attentes** : Le modèle doit renvoyer uniquement le bloc de code `{ ... }`. 
*   **Avertissement du Professeur** : Sans "Constrained Sampling" matériel (GBNF), le modèle peut parfois ajouter du texte ("Voici le JSON..."). 

> [!TIP]
🔑 **L'astuce d'ingénieur** : Si `json.loads` échoue, utilisez une expression régulière (Regex) pour extraire uniquement ce qui se trouve entre les accolades `{}`.

</details>
---

**Mots-clés de la semaine** : Persona, In-Context Learning, Few-shot, Chain-of-Thought (CoT), Self-Consistency, Tree-of-Thought, Lost in the Middle, Constrained Sampling, GBNF, JSON Formatting.

**En prévision de la semaine suivante** : Nous allons apprendre à combattre les hallucinations de manière radicale. Comment connecter votre IA à Internet ou à vos propres documents PDF pour qu'elle réponde avec des preuves ? Bienvenue dans le monde du **RAG (Retrieval-Augmented Generation)**.

**SOURCES COMPLÈTES** :
*   Guide de Prompting : [PromptingGuide.ai](https://www.promptingguide.ai/fr)
*   Blog Jay Alammar : *The Illustrated Transformer* (pour l'intuition de l'attention dans le CoT).